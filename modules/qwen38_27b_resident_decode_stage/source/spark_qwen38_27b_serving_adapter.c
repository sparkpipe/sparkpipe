
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>


static double clock_gettime_mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
	return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_memory_buffer.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen38_27b_model.h"
#include "spark_qwen38_27b_dspark_format.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen38_27b_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_speculation_seam.h"

#ifndef QWEN38_27B_MODEL_REVISION
#error "QWEN38_27B_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef QWEN38_27B_CONTRACT_SHA256
#error "QWEN38_27B_CONTRACT_SHA256 must identify the exact package contract"
#endif

#ifndef SPARK_QWEN38_27B_SERVING_TP_DEGREE
#define SPARK_QWEN38_27B_SERVING_TP_DEGREE 4u
#endif
#define SPARK_QWEN38_27B_SERVING_TP (SPARK_QWEN38_27B_SERVING_TP_DEGREE >= 1u)
#if SPARK_QWEN38_27B_SERVING_TP_DEGREE == 1u
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "spark.qwen38_27b.serving-adapter.tp1.v1"
#define SPARK_QWEN38_27B_SERVING_STAGE_COUNT 1u
#define SPARK_QWEN38_27B_SERVING_STAGE_LAYER_COUNTS {64u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u}
#else
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "spark.qwen38_27b.serving-adapter.tp4.v1"
#define SPARK_QWEN38_27B_SERVING_STAGE_COUNT 4u
#define SPARK_QWEN38_27B_SERVING_STAGE_LAYER_COUNTS {64u,64u,64u,64u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u}
#endif
#define SPARK_QWEN38_27B_SERVING_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN38_27B_SERVING_DRIVER_MODEL_ID \
	"alibaba.qwen3.8-27b.resident-decode-stage-firmware"
#define SPARK_QWEN38_27B_SERVING_STAGE_NAME "qwen38_27b_resident_decode_stage"
#define SPARK_QWEN38_27B_SERVING_TARGET \
	"cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
#define SPARK_QWEN38_27B_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_QWEN38_27B_SERVING_MAX_SEQUENCE_POSITIONS_CAP 262144u
#define SPARK_QWEN38_27B_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_QWEN38_27B_SERVING_SPECULATE_ENV "SPARK_QWEN38_27B_SERVING_SPECULATE"
#define SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_ENV "SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT"
#define SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV "SPARK_QWEN38_27B_SPECULATORS"
#define SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT 2u
#define SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_TIME_BUDGET_MS 20u
#define SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_MAX_DEPTH 16u
#define SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_MAX_NODE_COUNT 64u
#define SPARK_QWEN38_27B_SERVING_SEAM_CONNECT_TIMEOUT_MS 1000u
#define SPARK_QWEN38_27B_SERVING_SEAM_IO_TIMEOUT_MS 30000u
#define SPARK_QWEN38_27B_SERVING_AVAILABLE_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_MTP | \
	 SPARK_SPECULATION_SEAM_SOURCE_DSPARK | \
	 SPARK_SPECULATION_SEAM_SOURCE_DFLASH2 | \
	 SPARK_SPECULATION_SEAM_SOURCE_NGRAM | \
	 SPARK_SPECULATION_SEAM_SOURCE_SUFFIX | \
	 SPARK_SPECULATION_SEAM_SOURCE_NGRAM3)
#define SPARK_QWEN38_27B_SERVING_LOCAL_METHOD_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_MTP | \
	 SPARK_SPECULATION_SEAM_SOURCE_DSPARK | \
	 SPARK_SPECULATION_SEAM_SOURCE_DFLASH2)
#define SPARK_QWEN38_27B_SERVING_REMOTE_METHOD_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_NGRAM | \
	 SPARK_SPECULATION_SEAM_SOURCE_SUFFIX | \
	 SPARK_SPECULATION_SEAM_SOURCE_NGRAM3)

typedef struct SparkQwen38_27bServingState SparkQwen38_27bServingState;

static uint32_t SparkQwen38_27bServingSpeculativeDraftCount(const SparkQwen38_27bServingState *state);
#define SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_ENV "SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY"
#define SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_RECOVER 0u
#define SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT 1u
static uint32_t SparkQwen38_27bServingSpecFirstDraftPolicy(void)
{
	const char *value = getenv(SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_ENV);
	if ( value != 0 && strcmp(value,"strict") == 0 )
		return(SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT);
	return(SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_RECOVER);
}

#define SPARK_QWEN38_27B_SERVING_SPEC_METHOD_ENV "SPARK_QWEN38_27B_SERVING_SPEC_METHOD"
#define SPARK_QWEN38_27B_SERVING_SPEC_METHOD_MTP 0u
#define SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DSPARK 1u
#define SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DFLASH2 2u
#define SPARK_QWEN38_27B_SERVING_SPEC_METHOD_NONE 3u

static uint32_t SparkQwen38_27bServingBlockDraftMethod(uint32_t spec_method)
{
	return(spec_method == SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DSPARK || spec_method == SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DFLASH2);
}
static uint32_t SparkQwen38_27bServingActiveDraftCount(const SparkQwen38_27bServingState *state,uint32_t spec_method);
#define SPARK_QWEN38_27B_SERVING_GDN_SNAPSHOT_SLOTS \
	SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS
#define SPARK_QWEN38_27B_SERVING_MAX_COMMITTED_TOKENS \
	(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS + 2u)

static const char *const SparkQwen38_27bServingConfigurationMembers[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions"
};

static const char *const SparkQwen38_27bServingConfigurationMembersDraft[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"speculative_draft_count"
};

static const char *const SparkQwen38_27bServingConfigurationMembersBridge[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"draft_bridge_host",
	"draft_bridge_port"
};

static const char *const SparkQwen38_27bServingConfigurationMembersBridgeDraft[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"draft_bridge_host",
	"draft_bridge_port",
	"speculative_draft_count"
};

typedef struct SparkQwen38_27bServingSpecState
{
	uint32_t resident_slot;
	uint64_t base_position;
	uint64_t sequence_id;
	uint32_t snapshot_index;
	uint32_t draft_token_count;
	uint32_t accepted_count;
	uint32_t chain_dead;
	uint32_t first_draft_miss;
	uint32_t engine_staged;
	uint32_t draft_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	uint32_t emitted_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	uint32_t committed_ids[SPARK_QWEN38_27B_SERVING_MAX_COMMITTED_TOKENS];
} SparkQwen38_27bServingSpecState;

#define SPARK_QWEN38_27B_SERVING_EXTENSION_KIND 0x5136u
typedef struct SparkQwen38_27bServingSpecTelemetry
{
	uint32_t first_draft_miss_count;
	uint32_t first_draft_policy;
} SparkQwen38_27bServingSpecTelemetry;

typedef struct SparkQwen38_27bServingPending
{
	struct SparkQwen38_27bServingState *owner;
	SparkServingAdapterPendingCommon common;
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t dspark_draft_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	SparkQwen38_27bGdnSnapshotView prefix_gdn_view;
	SparkQwen38_27bServingSpecState spec[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t spec_active;
	uint32_t spec_tokens_per_sequence;
	uint32_t spec_total_accepted;
	uint32_t spec_chain_dead;
	uint32_t spec_first_draft_miss;
	uint32_t spec_fold;
} SparkQwen38_27bServingPending;

typedef struct SparkQwen38_27bServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkQwen38_27bServingTransportShim;

typedef struct SparkQwen38_27bServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
	void *execution_stream;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t stage_layer_count;
	uint32_t stage_attn_layer_count;
	uint32_t pipeline_slot_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t blocks_per_lane;
	uint32_t kv_block_count;
	uint32_t quiescing;
	uint32_t spec_method;
	uint32_t speculation_enabled;
	uint32_t speculative_draft_count;
	SparkSpeculationSeam *speculation_seam;
	char *bridge_host;
	uint32_t bridge_port;
	uint64_t orphan_completion_count;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkQwen38_27bKvBlockTableView block_table;
	SparkMemoryBuffer host_block_indices;
	SparkMemoryBuffer device_block_indices;
	SparkMemoryBuffer device_block_counts;
	SparkMemoryBuffer free_blocks;
	SparkMemoryBuffer block_refs;
	uint32_t free_block_count;
	uint32_t lane_prefix_entry[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_prefix_blocks[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_publish_identity[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT][32];
	uint32_t lane_publish_tokens[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_publish_armed[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_restore_slot[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_restore_armed[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	struct {
		uint8_t valid;
		uint8_t identity[32];
		uint32_t token_count;
		uint32_t block_count;
		uint32_t blocks[64];
		uint32_t refs;
		uint64_t last_used;
	} prefix_entries[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT];
	uint64_t prefix_epoch;
	uint32_t lane_block_counts[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkMemoryBuffer gather_scratch;
	SparkQwen38_27bServingTransportShim shim;
	uint32_t dflash2_drafts_valid;
	uint64_t dflash2_draft_sequence_id;
	uint32_t dflash2_next_draft_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	uint32_t dflash2_fold_armed;
	uint64_t dflash2_fold_position;
	uint64_t dflash2_fold_sequence_id;
	int32_t dflash2_fold_restore_slot;
	uint32_t dflash2_draft_matrix[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_MAX_MULTI_BLOCKS * (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE - 1u)];
	SparkQwen38_27bServingPending pending[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkQwen38_27bServingState;

static uint32_t SparkQwen38_27bServingSpeculativeDraftCount(const SparkQwen38_27bServingState *state)
{
	return(state->speculative_draft_count);
}

static uint32_t SparkQwen38_27bServingActiveDraftCount(const SparkQwen38_27bServingState *state,uint32_t spec_method)
{
	if ( !SparkQwen38_27bServingBlockDraftMethod(spec_method) )
		return(SparkQwen38_27bServingSpeculativeDraftCount(state));
	{
		uint32_t block = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE;
		uint32_t cap = SparkQwen38_27bServingSpeculativeDraftCount(state);
		return(cap < block ? cap : block);
	}
}

static const SparkModelServingAdapterDescriptor SparkQwen38_27bServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_QWEN38_27B_SERVING_ADAPTER_ID,
		SPARK_QWEN38_27B_SERVING_MODEL_ID,
		QWEN38_27B_MODEL_REVISION,
		SPARK_QWEN38_27B_SERVING_PROGRAM_NAME,
		QWEN38_27B_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION),
	.cache_block_token_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS,
	.stage_count = SPARK_QWEN38_27B_SERVING_STAGE_COUNT,
	.layer_count = SPARK_QWEN38_27B_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE,
	.stage_layer_counts = SPARK_QWEN38_27B_SERVING_STAGE_LAYER_COUNTS,
	.minimum_efficient_submission_row_count = 0u
};

#define SPARK_QWEN38_SERVING_ADAPTER_FN(name) SparkQwen38_27b##name
#define SPARK_QWEN38_SERVING_ADAPTER_TYPE(name) SparkQwen38_27b##name
#define SPARK_QWEN38_SERVING_ADAPTER_CONST(name) SPARK_QWEN38_27B_##name

#include "sparkpipe/spark_qwen38_serving_adapter_common.h"

static SparkStatus SparkQwen38_27bServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkQwen38_27bServingState *state,
	uint32_t *max_sequence_positions)
{
	SparkJsonDocument document;
	int32_t root,token;
	int32_t bridge_host_token,bridge_port_token,draft_count_token;
	uint32_t schema_version;
	char *relative_stage_pack_path;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	bridge_host_token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"draft_bridge_host") : -1;
	bridge_port_token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"draft_bridge_port") : -1;
	draft_count_token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"speculative_draft_count") : -1;
	if ( status == SPARK_STATUS_OK && (bridge_host_token < 0) != (bridge_port_token < 0) )
	{
		fprintf(stderr,"qwen38_27b_serving draft_bridge_host and draft_bridge_port must both be present or both absent\n");
		status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
	{
		const char *const *members = SparkQwen38_27bServingConfigurationMembers;
		uint32_t member_count = (uint32_t)(sizeof(SparkQwen38_27bServingConfigurationMembers) / sizeof(SparkQwen38_27bServingConfigurationMembers[0]));
		if ( bridge_host_token >= 0 && draft_count_token >= 0 )
		{
			members = SparkQwen38_27bServingConfigurationMembersBridgeDraft;
			member_count = (uint32_t)(sizeof(SparkQwen38_27bServingConfigurationMembersBridgeDraft) / sizeof(SparkQwen38_27bServingConfigurationMembersBridgeDraft[0]));
		}
		else if ( bridge_host_token >= 0 )
		{
			members = SparkQwen38_27bServingConfigurationMembersBridge;
			member_count = (uint32_t)(sizeof(SparkQwen38_27bServingConfigurationMembersBridge) / sizeof(SparkQwen38_27bServingConfigurationMembersBridge[0]));
		}
		else if ( draft_count_token >= 0 )
		{
			members = SparkQwen38_27bServingConfigurationMembersDraft;
			member_count = (uint32_t)(sizeof(SparkQwen38_27bServingConfigurationMembersDraft) / sizeof(SparkQwen38_27bServingConfigurationMembersDraft[0]));
		}
		status = SparkJsonValidateObjectMembersExact(&document,root,members,member_count);
	}
	state->speculative_draft_count = SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT;
	if ( status == SPARK_STATUS_OK && draft_count_token >= 0 )
	{
		status = SparkJsonGetUInt32(&document,draft_count_token,&state->speculative_draft_count);
		if ( status == SPARK_STATUS_OK && ( state->speculative_draft_count == 0u || state->speculative_draft_count > SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS ) )
		{
			fprintf(stderr,"qwen38_27b_serving speculative_draft_count must be 1..%u\n",(unsigned)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS);
			status = SPARK_STATUS_SCHEMA_ERROR;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_QWEN38_27B_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,QWEN38_27B_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
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
	return(status);
}

static uint32_t SparkQwen38_27bServingFirstLayer(uint32_t stage_index)
{
	uint32_t index,first_layer;
#if SPARK_QWEN38_27B_SERVING_TP
	(void)stage_index;
	return(0u);
#endif
	first_layer = 0u;
	for (index=0u; index<stage_index; index++)
		first_layer += SparkQwen38_27bServingDescriptor.stage_layer_counts[index];
	return(first_layer);
}

static uint32_t SparkQwen38_27bServingOwnsFinalHead(const SparkQwen38_27bServingState *state);

static SparkStatus SparkQwen38_27bServingRejectRetiredSpeculationEnvironment(void)
{
	if ( getenv(SPARK_QWEN38_27B_SERVING_SPECULATE_ENV) != 0 )
	{
		fprintf(stderr,"qwen38_27b_serving %s is retired: use %s (speculation source mask) instead\n",SPARK_QWEN38_27B_SERVING_SPECULATE_ENV,SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( getenv(SPARK_QWEN38_27B_SERVING_SPEC_METHOD_ENV) != 0 )
	{
		fprintf(stderr,"qwen38_27b_serving %s is retired: use %s (speculation source mask) instead\n",SPARK_QWEN38_27B_SERVING_SPEC_METHOD_ENV,SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( getenv(SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_ENV) != 0 )
	{
		fprintf(stderr,"qwen38_27b_serving %s is retired: use %s (speculation source mask) instead\n",SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_ENV,SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingResolveSpeculationMethods(
	uint32_t enabled_sources,
	uint32_t *spec_method_out,
	uint32_t *speculation_enabled_out)
{
	uint32_t local_sources;
	local_sources = enabled_sources & SPARK_QWEN38_27B_SERVING_LOCAL_METHOD_SOURCES;
	if ( (enabled_sources & SPARK_QWEN38_27B_SERVING_REMOTE_METHOD_SOURCES) != 0u && local_sources != 0u )
	{
		fprintf(stderr,"qwen38_27b_serving %s=0x%x selects remote sources together with a local method: not supported\n",SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV,enabled_sources);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( (enabled_sources & SPARK_QWEN38_27B_SERVING_REMOTE_METHOD_SOURCES) != 0u )
	{
		fprintf(stderr,"qwen38_27b_serving %s=0x%x selects remote drafting but this adapter does not track host-side committed token ids for the draft bridge\n",SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV,enabled_sources);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	if ( local_sources != 0u && (local_sources & (local_sources - 1u)) != 0u )
	{
		fprintf(stderr,"qwen38_27b_serving %s=0x%x selects more than one local speculation method\n",SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV,enabled_sources);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	*spec_method_out = (local_sources & SPARK_SPECULATION_SEAM_SOURCE_DSPARK) != 0u ? SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DSPARK :
		(local_sources & SPARK_SPECULATION_SEAM_SOURCE_DFLASH2) != 0u ? SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DFLASH2 :
		(local_sources & SPARK_SPECULATION_SEAM_SOURCE_MTP) != 0u ? SPARK_QWEN38_27B_SERVING_SPEC_METHOD_MTP :
		SPARK_QWEN38_27B_SERVING_SPEC_METHOD_NONE;
	*speculation_enabled_out = local_sources != 0u ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingInitializeSpeculationSeam(
	SparkQwen38_27bServingState *state)
{
	SparkSpeculationSeamConfiguration seam_configuration;
	const char *control_value;
	uint32_t available_sources;
	uint32_t enabled_sources;
	SparkStatus status;
	status = SparkQwen38_27bServingRejectRetiredSpeculationEnvironment();
	if ( status != SPARK_STATUS_OK )
		return(status);
	available_sources = SPARK_QWEN38_27B_SERVING_AVAILABLE_SOURCES;
	if ( state->bridge_host == 0 )
		available_sources &= ~SPARK_SPECULATION_SEAM_REMOTE_SOURCES;
	control_value = getenv(SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV);
	if ( control_value == 0 || (control_value[0] == '1' && control_value[1] == '\0') )
		enabled_sources = SPARK_SPECULATION_SEAM_SOURCE_MTP & available_sources;
	else
	{
		status = SparkSpeculationSeamParseControl(control_value,available_sources,&enabled_sources);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"qwen38_27b_serving %s control value rejected: status=%d available=0x%x\n",SPARK_QWEN38_27B_SERVING_SPECULATORS_ENV,(int)status,available_sources);
			return(status);
		}
	}
	status = SparkQwen38_27bServingResolveSpeculationMethods(enabled_sources,&state->spec_method,&state->speculation_enabled);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&seam_configuration,0,sizeof(seam_configuration));
	seam_configuration.abi_version = SPARK_SPECULATION_SEAM_ABI_VERSION;
	seam_configuration.descriptor_bytes = SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES;
	seam_configuration.available_source_mask = available_sources;
	seam_configuration.default_source_mask = SPARK_SPECULATION_SEAM_SOURCE_MTP & available_sources;
	seam_configuration.default_speculative_token_count = state->speculative_draft_count;
	seam_configuration.lane_count = state->max_active_sequence_count;
	seam_configuration.max_committed_token_count = state->max_sequence_positions;
	seam_configuration.max_tap_row_count = 0u;
	seam_configuration.draft_time_budget_ms = SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_TIME_BUDGET_MS;
	seam_configuration.draft_max_depth = SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_MAX_DEPTH;
	seam_configuration.draft_max_node_count = SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_MAX_NODE_COUNT;
	seam_configuration.connect_timeout_ms = SPARK_QWEN38_27B_SERVING_SEAM_CONNECT_TIMEOUT_MS;
	seam_configuration.io_timeout_ms = SPARK_QWEN38_27B_SERVING_SEAM_IO_TIMEOUT_MS;
	seam_configuration.control_value = control_value;
	seam_configuration.bridge_host = state->bridge_host;
	seam_configuration.bridge_port = state->bridge_port;
	memcpy(seam_configuration.target_model,SPARK_QWEN38_27B_SERVING_MODEL_ID,sizeof(SPARK_QWEN38_27B_SERVING_MODEL_ID));
	seam_configuration.model_contract.abi_version = SPARK_SPECULATION_ABI_VERSION;
	seam_configuration.model_contract.descriptor_bytes = SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES;
	seam_configuration.model_contract.verifier_hidden_dtype = SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16;
	seam_configuration.model_contract.draft_dtype = SPARK_SPECULATION_DRAFT_DTYPE_BF16;
	seam_configuration.model_contract.draft_layer_count = SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT;
	seam_configuration.model_contract.block_size = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE;
	seam_configuration.model_contract.hidden_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	seam_configuration.model_contract.intermediate_dimension = SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION;
	seam_configuration.model_contract.attention_head_count = SPARK_QWEN38_27B_MODEL_ATTENTION_HEAD_COUNT;
	seam_configuration.model_contract.kv_head_count = SPARK_QWEN38_27B_MODEL_KV_HEAD_COUNT;
	seam_configuration.model_contract.head_dimension = SPARK_QWEN38_27B_MODEL_HEAD_DIMENSION;
	seam_configuration.model_contract.vocab_size = SPARK_QWEN38_27B_MODEL_VOCAB_COUNT;
	seam_configuration.model_contract.draft_vocab_size = SPARK_QWEN38_27B_MODEL_VOCAB_COUNT;
	seam_configuration.model_contract.maximum_speculative_token_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS;
	seam_configuration.model_contract.verifier_accept_k = 1u;
	status = SparkSpeculationSeamInitialize(&seam_configuration,&state->speculation_seam);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"qwen38_27b_serving speculation seam init failed: status=%d\n",(int)status);
		return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingSetEnvironment(
	const SparkQwen38_27bServingState *state)
{
	char value[32];
#define SPARK_QWEN38_27B_SERVING_SET_TEXT(name,text) \
	do { if ( setenv(name,text,1) != 0 ) return(SPARK_STATUS_INTERNAL_ERROR); } while (0)
#define SPARK_QWEN38_27B_SERVING_SET_UNSIGNED(name,number) \
	do { snprintf(value,sizeof(value),"%u",(uint32_t)(number)); SPARK_QWEN38_27B_SERVING_SET_TEXT(name,value); } while (0)
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_ALLOW_UNQUALIFIED_EXECUTION","1");
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_PACK_PATH",state->stage_pack_path);
#if SPARK_QWEN38_27B_SERVING_TP
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_COUNT",1u);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_INDEX",0u);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_FIRST_LAYER",0u);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_LAYER_COUNT",SPARK_QWEN38_27B_MODEL_LAYER_COUNT);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_TP_DEGREE",SPARK_QWEN38_27B_SERVING_TP_DEGREE);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_TP_RANK",state->stage_index);
#else
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_COUNT",SPARK_QWEN38_27B_SERVING_STAGE_COUNT);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_INDEX",state->stage_index);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_FIRST_LAYER",state->first_layer_index);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_LAYER_COUNT",state->stage_layer_count);
#endif
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_MAX_ACTIVE_SEQUENCES",state->max_active_sequence_count);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_MAX_INPUT_ROWS",state->max_input_row_count);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_PIPELINE_SLOTS",state->pipeline_slot_count);
	SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_KV_BLOCKS",state->kv_block_count);
	if ( state->speculation_enabled != 0u && SparkQwen38_27bServingOwnsFinalHead(state) != 0u )
	{
		SPARK_QWEN38_27B_SERVING_SET_UNSIGNED("SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS",SparkQwen38_27bServingBlockDraftMethod(state->spec_method) ? 16u : SPARK_QWEN38_27B_SERVING_GDN_SNAPSHOT_SLOTS);
		if ( !SparkQwen38_27bServingBlockDraftMethod(state->spec_method) )
			SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_MTP","1");
		else
			SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_MTP","0");
	}
	else
	{
		SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_MTP","0");
		SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS","0");
	}
	if ( state->speculation_enabled != 0u && SparkQwen38_27bServingOwnsFinalHead(state) != 0u && SparkQwen38_27bServingBlockDraftMethod(state->spec_method) != 0u )
	{
		const char *drafter_pack = getenv("SPARK_QWEN38_27B_DSPARK_PACK_PATH");
		if ( drafter_pack == 0 || drafter_pack[0] == '\0' )
		{
			fprintf(stderr,"qwen38_27b_serving spec method requires SPARK_QWEN38_27B_DSPARK_PACK_PATH\n");
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	}
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_KV_STORE","none");
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_KV_SERVICE","none");
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_KV_SOCKET","none");
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_KV_POOL_BYTES","0");
	SPARK_QWEN38_27B_SERVING_SET_TEXT("SPARK_QWEN38_27B_STAGE_KV_WORKERS","0");
#undef SPARK_QWEN38_27B_SERVING_SET_TEXT
#undef SPARK_QWEN38_27B_SERVING_SET_UNSIGNED
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingRowOrderReject(
	const SparkModelServingSubmission *submission,
	const char *reason)
{
	fprintf(stderr, "qwen38_27b_roworder_reject reason=%s kind=%u rows=%u lanes=%u pos=%llu slot=%u\n",
		reason, submission->work_kind, submission->row_count, submission->active_sequence_count,
		(unsigned long long)(submission->row_count != 0u ? submission->row_positions[0] : 0u),
		submission->active_sequence_count != 0u ? submission->lanes[0].resident_sequence_slot : 0u);
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkQwen38_27bServingValidateRowOrder(
	const SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint8_t slot_seen[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->resident_sequence_capacity || slot_seen[slot] != 0u )
			return(SparkQwen38_27bServingRowOrderReject(submission,"slot_range_or_dup"));
		slot_seen[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
			return(SparkQwen38_27bServingRowOrderReject(submission,"lane_or_position_range"));
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SparkQwen38_27bServingRowOrderReject(submission,"row_position_gap"));
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SparkQwen38_27bServingRowOrderReject(submission,"decode_row_count"));
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SparkQwen38_27bServingRowOrderReject(submission,"wave_order"));
	return(row == submission->row_count ? SPARK_STATUS_OK : SparkQwen38_27bServingRowOrderReject(submission,"row_count_mismatch"));
}

static uint32_t SparkQwen38_27bServingOwnsEmbedding(const SparkQwen38_27bServingState *state)
{
	(void)state;
#if SPARK_QWEN38_27B_SERVING_TP
	return(1u);
#else
	return(state->stage_index == 0u ? 1u : 0u);
#endif
}

static uint32_t SparkQwen38_27bServingOwnsFinalHead(const SparkQwen38_27bServingState *state)
{
	(void)state;
#if SPARK_QWEN38_27B_SERVING_TP
	return(1u);
#else
	return(state->stage_index + 1u == SPARK_QWEN38_27B_SERVING_STAGE_COUNT ? 1u : 0u);
#endif
}

static uint32_t SparkQwen38_27bServingNeedsHiddenOutput(const SparkQwen38_27bServingState *state)
{
	(void)state;
#if SPARK_QWEN38_27B_SERVING_TP
	return(0u);
#else
	return(state->stage_index + 1u < SPARK_QWEN38_27B_SERVING_STAGE_COUNT ? 1u : 0u);
#endif
}

static SparkStatus SparkQwen38_27bServingValidateBoundaries(
	const SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES;
	if ( (SparkQwen38_27bServingOwnsEmbedding(state) == 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (SparkQwen38_27bServingOwnsEmbedding(state) != 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (SparkQwen38_27bServingNeedsHiddenOutput(state) != 0u && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (SparkQwen38_27bServingNeedsHiddenOutput(state) == 0u && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingValidateSubmissionBase(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmissionPrevalidated(&SparkQwen38_27bServingDescriptor,&state->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"qwen38_27b_debug validate_runtime status=%d kind=%u rows=%u lanes=%u act=%u tps=%u new_tokens=%u pos=%llu ctx=%llu\\n",(int)status,submission->work_kind,submission->row_count,submission->lane_count,submission->active_sequence_count,submission->tokens_per_sequence,submission->new_token_count,(unsigned long long)submission->sequence_position,(unsigned long long)(submission->active_sequence_count > 0u ? submission->lanes[0].context_token_count : 0u));
		return(status);
	}
	if ( submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkQwen38_27bServingValidateRowOrder(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"qwen38_27b_debug row_order status=%d\\n",(int)status);
		return(status);
	}
	if ( submission->model_extension_bytes != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen38_27bServingState *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SparkQwen38_27bServingState *)adapter_state;
	status = SparkQwen38_27bServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}

static void SparkQwen38_27bServingBlockRelease(SparkQwen38_27bServingState *state, uint32_t block)
{
	if ( state->block_refs.pointer != 0 && --((uint16_t *)state->block_refs.pointer)[block] == 0u )
		((uint32_t *)state->free_blocks.pointer)[state->free_block_count++] = block;
}

static void SparkQwen38_27bServingReleaseLane(
	SparkQwen38_27bServingState *state,
	uint32_t slot)
{
	uint32_t ordinal;
	for (ordinal=0u; ordinal<state->lane_block_counts[slot]; ordinal++)
		SparkQwen38_27bServingBlockRelease(state,((uint32_t *)state->host_block_indices.pointer)[((uint64_t)slot * state->blocks_per_lane) + ordinal]);
	if ( state->lane_prefix_entry[slot] != 0xFFu )
	{
		state->prefix_entries[state->lane_prefix_entry[slot]].refs--;
		state->lane_prefix_entry[slot] = 0xFFu;
	}
	state->lane_prefix_blocks[slot] = 0u;
	state->lane_block_counts[slot] = 0u;
	state->lane_context_tokens[slot] = 0u;
	state->lane_publish_armed[slot] = 0u;
	state->lane_restore_armed[slot] = 0u;
}

static SparkStatus SparkQwen38_27bServingCoverLane(
	SparkQwen38_27bServingState *state,
	uint32_t slot,
	uint64_t end_position)
{
	uint32_t required,ordinal;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	required = (uint32_t)((end_position + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	if ( required > state->blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (ordinal=state->lane_block_counts[slot]; ordinal<required; ordinal++)
	{
		if ( state->free_block_count == 0u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		((uint32_t *)state->host_block_indices.pointer)[((uint64_t)slot * state->blocks_per_lane) + ordinal] = ((uint32_t *)state->free_blocks.pointer)[--state->free_block_count];
		if ( state->block_refs.pointer != 0 )
			((uint16_t *)state->block_refs.pointer)[((uint32_t *)state->host_block_indices.pointer)[((uint64_t)slot * state->blocks_per_lane) + ordinal]] = 1u;
		state->lane_block_counts[slot] = ordinal + 1u;
	}
	state->lane_block_counts[slot] = required;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkQwen38_27bServingPrefixFind(SparkQwen38_27bServingState *state, const uint8_t *identity, uint32_t token_count)
{
	uint32_t index;
	for (index=0u; index<SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT; index++)
		if ( state->prefix_entries[index].valid != 0u && state->prefix_entries[index].token_count == token_count &&
			memcmp(state->prefix_entries[index].identity,identity,32) == 0 )
			return(index);
	return(0xFFu);
}

static SparkStatus SparkQwen38_27bServingPrefixPublish(SparkQwen38_27bServingState *state, uint32_t slot)
{
	uint32_t index,blocks,ordinal,free_index;
	uint64_t used;
	blocks = (state->lane_publish_tokens[slot] + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	if ( blocks == 0u || blocks > 64u || blocks > state->lane_block_counts[slot] )
		return(SPARK_STATUS_OK);
	index = SparkQwen38_27bServingPrefixFind(state,state->lane_publish_identity[slot],state->lane_publish_tokens[slot]);
	if ( index == 0xFFu )
	{
		free_index = 0xFFu;
		used = 0u;
		for (index=0u; index<SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT; index++)
		{
			if ( state->prefix_entries[index].valid == 0u )
			{
				free_index = index;
				break;
			}
			if ( state->prefix_entries[index].refs == 0u && (free_index == 0xFFu || state->prefix_entries[index].last_used < used) )
			{
				used = state->prefix_entries[index].last_used;
				free_index = index;
			}
		}
		index = free_index;
		if ( index == 0xFFu )
			return(SPARK_STATUS_OK);
		if ( state->prefix_entries[index].valid != 0u )
		{
			for (ordinal=0u; ordinal<state->prefix_entries[index].block_count; ordinal++)
				SparkQwen38_27bServingBlockRelease(state,state->prefix_entries[index].blocks[ordinal]);
			memset(&state->prefix_entries[index],0,sizeof(state->prefix_entries[index]));
		}
		memcpy(state->prefix_entries[index].identity,state->lane_publish_identity[slot],32);
		state->prefix_entries[index].valid = 1u;
		state->prefix_entries[index].token_count = state->lane_publish_tokens[slot];
		state->prefix_entries[index].block_count = blocks;
		for (ordinal=0u; ordinal<blocks; ordinal++)
			state->prefix_entries[index].blocks[ordinal] = ((uint32_t *)state->host_block_indices.pointer)[((uint64_t)slot * state->blocks_per_lane) + ordinal];
		state->prefix_entries[index].refs = 0u;
		for (ordinal=0u; ordinal<blocks; ordinal++)
			if ( state->block_refs.pointer != 0 )
				((uint16_t *)state->block_refs.pointer)[state->prefix_entries[index].blocks[ordinal]]++;
	}
	state->prefix_entries[index].last_used = ++state->prefix_epoch;
	if ( state->lane_prefix_entry[slot] != index )
	{
		uint32_t previous = state->lane_prefix_entry[slot];
		if ( previous != 0xFFu && state->prefix_entries[previous].refs != 0u )
			state->prefix_entries[previous].refs--;
		state->prefix_entries[index].refs++;
		state->lane_prefix_entry[slot] = index;
		state->lane_prefix_blocks[slot] = state->prefix_entries[index].block_count;
	}
	fprintf(stderr,"qwen38_27b_prefix publish slot=%u entry=%u tokens=%u blocks=%u\n",slot,index,state->lane_publish_tokens[slot],blocks);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingPrefixBorrow(SparkQwen38_27bServingState *state, uint32_t slot, const uint8_t *identity, uint32_t token_count)
{
	uint32_t index,blocks,ordinal;
	index = SparkQwen38_27bServingPrefixFind(state,identity,token_count);
	if ( index == 0xFFu )
		return(SPARK_STATUS_NOT_FOUND);
	blocks = state->prefix_entries[index].block_count;
	if ( blocks > state->blocks_per_lane )
		return(SPARK_STATUS_NOT_FOUND);
	for (ordinal=0u; ordinal<blocks; ordinal++)
	{
		uint32_t block = state->prefix_entries[index].blocks[ordinal];
		((uint32_t *)state->host_block_indices.pointer)[((uint64_t)slot * state->blocks_per_lane) + ordinal] = block;
		if ( state->block_refs.pointer != 0 )
			((uint16_t *)state->block_refs.pointer)[block]++;
	}
	state->lane_block_counts[slot] = blocks;
	state->lane_context_tokens[slot] = token_count;
	state->lane_prefix_entry[slot] = index;
	state->lane_prefix_blocks[slot] = blocks;
	state->prefix_entries[index].refs++;
	state->prefix_entries[index].last_used = ++state->prefix_epoch;
	state->lane_restore_slot[slot] = index;
	state->lane_restore_armed[slot] = 1u;
	fprintf(stderr,"qwen38_27b_prefix borrow slot=%u entry=%u tokens=%u blocks=%u\n",slot,index,token_count,blocks);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingCoverSubmission(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	SparkStatus status;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		uint64_t first_position,end_position;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u && state->lane_restore_armed[slot] == 0u )
		{
			SparkStatus borrow = SparkQwen38_27bServingPrefixBorrow(state,slot,submission->lanes[lane].cache_prefix_identity.sha256,submission->lanes[lane].cache_prefix_token_count);
			if ( borrow != SPARK_STATUS_OK )
			{
				fprintf(stderr,"qwen38_27b_prefix miss slot=%u tokens=%u - recomputing\n",slot,submission->lanes[lane].cache_prefix_token_count);
				state->lane_restore_armed[slot] = 0u;
			}
		}
		if ( (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u )
		{
			memcpy(state->lane_publish_identity[slot],submission->lanes[lane].cache_publish_identity.sha256,32);
			state->lane_publish_tokens[slot] = submission->lanes[lane].cache_publish_token_count;
			state->lane_publish_armed[slot] = 1u;
		}
		first_position = UINT64_MAX;
		end_position = 0u;
		for (row=0u; row<submission->row_count; row++)
		{
			if ( submission->row_lane_indices[row] != lane )
				continue;
			if ( submission->row_positions[row] < first_position )
				first_position = submission->row_positions[row];
			if ( submission->row_positions[row] + 1u > end_position )
				end_position = submission->row_positions[row] + 1u;
		}
		if ( first_position == 0u && state->lane_context_tokens[slot] != 0u )
			SparkQwen38_27bServingReleaseLane(state,slot);
		status = SparkQwen38_27bServingCoverLane(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
		{
			SparkQwen38_27bServingDropSubmission(state,submission);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen38_27bServingCommitSubmission(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	const SparkQwen38_27bServingPending *pending)
{
	uint32_t lane,row;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( pending->spec_active != 0u )
		{
			uint32_t last_row;
			last_row = pending->last_row_by_lane[lane];
			if ( submission->row_positions[last_row] + pending->spec_tokens_per_sequence > state->lane_context_tokens[slot] )
				state->lane_context_tokens[slot] = submission->row_positions[last_row] + pending->spec_tokens_per_sequence;
		}
		else
		{
			for (row=0u; row<submission->row_count; row++)
				if ( submission->row_lane_indices[row] == lane && submission->row_positions[row] + 1u > state->lane_context_tokens[slot] )
					state->lane_context_tokens[slot] = submission->row_positions[row] + 1u;
		}
	}
}

static SparkStatus SparkQwen38_27bServingUploadBlockTable(
	const SparkQwen38_27bServingState *state)
{
	SparkMemoryBuffer destination;
	SparkMemoryBuffer host_indices;
	SparkMemoryBuffer host_counts;
	uint64_t indices_bytes,counts_bytes;
	SparkStatus status;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	indices_bytes = (uint64_t)state->max_active_sequence_count * state->blocks_per_lane * sizeof(uint32_t);
	counts_bytes = (uint64_t)state->max_active_sequence_count * sizeof(uint32_t);
	host_indices = SPARK_MEMORY_BUFFER_VIEW(state->host_block_indices.pointer,
		SPARK_MEMORY_SPACE_HOST_COHERENT,indices_bytes);
	host_counts = SPARK_MEMORY_BUFFER_VIEW((uint32_t *)state->lane_block_counts,
		SPARK_MEMORY_SPACE_HOST_COHERENT,counts_bytes);
	destination = state->device_block_indices;
	status = SparkMemoryBufferCopy(&destination,&host_indices,indices_bytes,0);
	if ( status == SPARK_STATUS_OK )
	{
		destination = state->device_block_counts;
		status = SparkMemoryBufferCopy(&destination,&host_counts,counts_bytes,0);
	}
	return(status);
}

static SparkStatus SparkQwen38_27bServingExtendSpeculativeCoverage(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	uint64_t total_needed;
	total_needed = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		uint64_t position,end_position,required;
		slot = submission->lanes[lane].resident_sequence_slot;
		position = 0u;
		for (row=0u; row<submission->row_count; row++)
			if ( submission->row_lane_indices[row] == lane )
				position = submission->row_positions[row];
		end_position = position + (uint64_t)SparkQwen38_27bServingActiveDraftCount(state,state->spec_method) + 2u;
		required = (end_position + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		if ( required > state->blocks_per_lane )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( required > state->lane_block_counts[slot] )
			total_needed += required - state->lane_block_counts[slot];
	}
	if ( total_needed > state->free_block_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		uint64_t position,end_position;
		SparkStatus status;
		slot = submission->lanes[lane].resident_sequence_slot;
		position = 0u;
		for (row=0u; row<submission->row_count; row++)
			if ( submission->row_lane_indices[row] == lane )
				position = submission->row_positions[row];
		end_position = position + (uint64_t)SparkQwen38_27bServingActiveDraftCount(state,state->spec_method) + 2u;
		status = SparkQwen38_27bServingCoverLane(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen38_27bServingBuildFrame(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38_27bServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows,
	SparkQwen38_27bDecodeBatchView *decode_batch,
	SparkQwen38_27bPrefillFrameView *prefill_view,
	SparkQwen38_27bResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t slot;
	uint64_t base_position;
	uint32_t row;
	slot = prefill != 0u ? pending->resident_slots[lane] : 0u;
	base_position = 0u;
	memset(decode_batch,0,sizeof(*decode_batch));
	memset(prefill_view,0,sizeof(*prefill_view));
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
		context->kv_block_table = &state->block_table;
	}
	if ( SparkQwen38_27bServingOwnsEmbedding(state) == 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen38_27bServingPostReceive;
	}
	if ( SparkQwen38_27bServingNeedsHiddenOutput(state) != 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SparkQwen38_27bServingSend;
	}
	state->shim.input_base = submission->hidden_input_address;
	state->shim.input_rows = frame_rows;
	state->shim.output_base = submission->hidden_output_address;
	if ( prefill != 0u )
	{
		uint32_t lane_row,flat;
		lane_row = 0u;
		for (flat=0u; flat<submission->row_count; flat++)
		{
			if ( submission->row_lane_indices[flat] != lane )
				continue;
			if ( lane_row >= wave_base && lane_row < wave_base + frame_rows )
			{
				pending->frame_row_flats[lane_row - wave_base] = flat;
				pending->frame_token_ids[lane_row - wave_base] = submission->token_ids[flat];
			}
			lane_row++;
		}
		state->shim.input_row_map = pending->frame_row_flats;
		state->shim.output_row_map = pending->frame_row_flats;
		base_position = submission->row_positions[pending->frame_row_flats[0]];
		prefill_view->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = submission->row_sequence_ids[pending->frame_row_flats[0]];
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW;
		context->prefill_frame = prefill_view;
		if ( state->lane_restore_armed[slot] != 0u && base_position == state->lane_context_tokens[slot] )
		{
			memset(&pending->prefix_gdn_view,0,sizeof(pending->prefix_gdn_view));
			pending->prefix_gdn_view.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
			pending->prefix_gdn_view.descriptor_bytes = sizeof(pending->prefix_gdn_view);
			pending->prefix_gdn_view.snapshot_index = state->lane_restore_slot[slot];
			context->gdn_snapshot = &pending->prefix_gdn_view;
			context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN;
			state->lane_restore_armed[slot] = 0u;
		}
		else if ( state->lane_publish_armed[slot] != 0u && base_position + frame_rows == state->lane_publish_tokens[slot] )
		{
			SparkStatus publish_status = SparkQwen38_27bServingPrefixPublish(state,slot);
			if ( publish_status == SPARK_STATUS_OK && state->lane_prefix_entry[slot] != 0xFFu )
			{
				memset(&pending->prefix_gdn_view,0,sizeof(pending->prefix_gdn_view));
				pending->prefix_gdn_view.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
				pending->prefix_gdn_view.descriptor_bytes = sizeof(pending->prefix_gdn_view);
				pending->prefix_gdn_view.snapshot_index = state->lane_prefix_entry[slot];
				context->gdn_snapshot = &pending->prefix_gdn_view;
				context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_SNAPSHOT_OUT;
				state->lane_publish_armed[slot] = 0u;
			}
		}
	}
	else
	{
		for (row=0u; row<frame_rows; row++)
			pending->frame_row_slots[row] = pending->resident_slots[submission->row_lane_indices[row]];
		memcpy(pending->frame_token_ids,submission->token_ids,(size_t)frame_rows * sizeof(uint32_t));
		state->shim.input_row_map = 0;
		state->shim.output_row_map = 0;
		decode_batch->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = submission->row_positions;
		decode_batch->row_sequence_ids = submission->row_sequence_ids;
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
		context->decode_batch = decode_batch;
	}
	memset(buffers,0,sizeof(SparkModelDriverBuffer[2]));
	if ( SparkQwen38_27bServingOwnsEmbedding(state) != 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen38_27bServingOwnsFinalHead(state) != 0u )
	{
		uint32_t out_index;
		out_index = SparkQwen38_27bServingOwnsEmbedding(state) != 0u ? 1u : 0u;
		buffers[out_index].slot = 1u;
		buffers[out_index].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffers[out_index].address = pending->frame_output_ids;
		buffers[out_index].bytes = (uint64_t)(prefill != 0u ? 1u : frame_rows) * sizeof(uint32_t);
	}
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = prefill != 0u ? prefill_view->sequence_id : submission->sequence_id;
	frame->sequence_position = prefill != 0u ? base_position : submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = prefill != 0u ? 1u : submission->active_sequence_count;
	frame->new_token_count = frame_rows;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = SparkQwen38_27bServingOwnsEmbedding(state) != 0u || SparkQwen38_27bServingOwnsFinalHead(state) != 0u ? buffers : 0;
	frame->buffer_count = (SparkQwen38_27bServingOwnsEmbedding(state) != 0u ? 1u : 0u) + (SparkQwen38_27bServingOwnsFinalHead(state) != 0u ? 1u : 0u);
	frame->residency = submission->residency;
	frame->scalar[0] = submission->request_generation;
	frame->user_context = context;
	frame->completion_function = SparkQwen38_27bServingDriverCompletion;
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SparkQwen38_27bServingRunFrame(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38_27bServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows)
{
	SparkQwen38_27bDecodeBatchView decode_batch;
	SparkQwen38_27bPrefillFrameView prefill_view;
	SparkQwen38_27bResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SparkQwen38_27bServingBuildFrame(state,submission,pending,prefill,lane,wave_base,frame_rows,&decode_batch,&prefill_view,&context,buffers,&frame);
	fprintf(stderr, "qwen38_27b_debug frame_ctx prefill=%u rows=%u ctx_flags=0x%x kv_table=%p row_pos0=%llu row_seq0=%llu row_slot0=%u buf0=%p/%llu/0x%x buf1=%p/%llu/0x%x slot1=%u\n",
		prefill, frame_rows, context.flags, (const void *)context.kv_block_table,
		(unsigned long long)(prefill == 0u ? decode_batch.row_positions[0] : 0ull),
		(unsigned long long)(prefill == 0u ? decode_batch.row_sequence_ids[0] : 0ull),
		(unsigned)(prefill == 0u ? decode_batch.row_lane_indices[0] : 0u),
		(const void *)buffers[0].address, (unsigned long long)buffers[0].bytes, buffers[0].flags,
		(const void *)buffers[1].address, (unsigned long long)buffers[1].bytes, buffers[1].flags,
		buffers[1].slot);
	status = SparkQwen38_27bServingAdmit(state,submission,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen38_27b_admit_reject status=%d prefill=%u frame_rows=%u lanes=%u\n", (int)status, prefill, frame_rows, submission->active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen38_27b_debug driver_submit status=%d prefill=%u rows=%u seqpos=%llu tps=%u newtok=%u\n", (int)status, prefill, frame_rows, (unsigned long long)submission->sequence_position, submission->tokens_per_sequence, submission->new_token_count);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen38_27b_debug frame_status status=%d prefill=%u rows=%u\n", (int)status, prefill, frame_rows);
	if ( status == SPARK_STATUS_OK && SparkQwen38_27bServingOwnsFinalHead(state) != 0u )
	{
		if ( prefill != 0u )
			pending->output_token_ids[lane] = (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u ? pending->frame_output_ids[0] : 0u;
		else
		{
			uint32_t row;
			for (row=0u; row<frame_rows; row++)
				pending->output_token_ids[submission->row_lane_indices[row]] = pending->frame_output_ids[row];
		}
	}
	return(status);
}

static void SparkQwen38_27bServingBuildSpeculativeFrame(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38_27bServingPending *pending,
	uint32_t slot,
	uint32_t prefill,
	const uint32_t *token_ids,
	const uint64_t *row_positions,
	const uint64_t *row_sequence_ids,
	uint32_t frame_rows,
	uint64_t base_position,
	uint64_t frame_sequence_id,
	uint64_t frame_sequence_position,
	uint32_t extra_flags,
	SparkQwen38_27bMtpDraftView *mtp_draft,
	SparkQwen38_27bDsparkDraftView *dspark_draft,
	SparkQwen38_27bGdnSnapshotView *gdn_snapshot,
	uint32_t output_id_count,
	SparkQwen38_27bDecodeBatchView *decode_batch,
	SparkQwen38_27bPrefillFrameView *prefill_view,
	SparkQwen38_27bResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t out_index;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
		context->kv_block_table = &state->block_table;
	}
	if ( SparkQwen38_27bServingOwnsEmbedding(state) == 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen38_27bServingPostReceive;
	}
	if ( SparkQwen38_27bServingNeedsHiddenOutput(state) != 0u )
	{
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SparkQwen38_27bServingSend;
	}
	state->shim.input_base = submission->hidden_input_address;
	state->shim.input_rows = frame_rows;
	state->shim.input_row_map = 0;
	state->shim.output_base = submission->hidden_output_address;
	state->shim.output_row_map = 0;
	memcpy(pending->frame_token_ids,token_ids,(size_t)frame_rows * sizeof(uint32_t));
	if ( prefill != 0u )
	{
		prefill_view->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = frame_sequence_id;
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW;
		context->prefill_frame = prefill_view;
	}
	else
	{
		pending->frame_row_slots[0] = slot;
		decode_batch->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = row_positions;
		decode_batch->row_sequence_ids = row_sequence_ids;
		context->flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
		context->decode_batch = decode_batch;
	}
	context->flags |= extra_flags;
	if ( mtp_draft != 0 )
		context->mtp_draft = mtp_draft;
	if ( dspark_draft != 0 )
		context->dspark_draft = dspark_draft;
	if ( gdn_snapshot != 0 )
		context->gdn_snapshot = gdn_snapshot;
	memset(buffers,0,sizeof(SparkModelDriverBuffer[2]));
	if ( SparkQwen38_27bServingOwnsEmbedding(state) != 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen38_27bServingOwnsFinalHead(state) != 0u )
	{
		out_index = SparkQwen38_27bServingOwnsEmbedding(state) != 0u ? 1u : 0u;
		buffers[out_index].slot = 1u;
		buffers[out_index].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffers[out_index].address = pending->frame_output_ids;
		buffers[out_index].bytes = (uint64_t)output_id_count * sizeof(uint32_t);
	}
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = frame_sequence_id;
	frame->sequence_position = frame_sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = 1u;
	frame->new_token_count = frame_rows;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = SparkQwen38_27bServingOwnsEmbedding(state) != 0u || SparkQwen38_27bServingOwnsFinalHead(state) != 0u ? buffers : 0;
	frame->buffer_count = (SparkQwen38_27bServingOwnsEmbedding(state) != 0u ? 1u : 0u) + (SparkQwen38_27bServingOwnsFinalHead(state) != 0u ? 1u : 0u);
	frame->residency = submission->residency;
	frame->scalar[0] = submission->request_generation;
	frame->user_context = context;
	frame->completion_function = SparkQwen38_27bServingDriverCompletion;
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SparkQwen38_27bServingRunSpeculativeFrame(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38_27bServingPending *pending,
	uint32_t slot,
	uint32_t prefill,
	const uint32_t *token_ids,
	const uint64_t *row_positions,
	const uint64_t *row_sequence_ids,
	uint32_t frame_rows,
	uint64_t base_position,
	uint64_t frame_sequence_id,
	uint64_t frame_sequence_position,
	uint32_t extra_flags,
	SparkQwen38_27bMtpDraftView *mtp_draft,
	SparkQwen38_27bDsparkDraftView *dspark_draft,
	SparkQwen38_27bGdnSnapshotView *gdn_snapshot,
	uint32_t output_id_count)
{
	SparkQwen38_27bDecodeBatchView decode_batch;
	SparkQwen38_27bPrefillFrameView prefill_view;
	SparkQwen38_27bResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SparkQwen38_27bServingBuildSpeculativeFrame(state,submission,pending,slot,prefill,token_ids,row_positions,row_sequence_ids,frame_rows,base_position,frame_sequence_id,frame_sequence_position,extra_flags,mtp_draft,dspark_draft,gdn_snapshot,output_id_count,&decode_batch,&prefill_view,&context,buffers,&frame);
	status = SparkQwen38_27bServingAdmit(state,submission,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen38_27b_admit_reject status=%d prefill=%u frame_rows=%u lanes=%u\n", (int)status, prefill, frame_rows, submission->active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	return(status);
}

static SparkStatus SparkQwen38_27bServingSubmitSpeculativeDecode(
	SparkQwen38_27bServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38_27bServingPending *pending)
{
	SparkQwen38_27bMtpDraftView mtp_draft;
	SparkQwen38_27bDsparkDraftView dspark_draft;
	SparkQwen38_27bGdnSnapshotView gdn_snapshot;
	SparkSpeculationPolicyVerifyResult verify_result;
	uint32_t lane,draft;
	uint32_t draft_count;
	uint32_t min_accepted;
	uint32_t first_draft_policy;
	uint32_t spec_method;
	uint32_t verify_tokens[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	SparkStatus status;
	spec_method = state->spec_method;
	draft_count = SparkQwen38_27bServingActiveDraftCount(state,state->spec_method);
	first_draft_policy = SparkQwen38_27bServingSpecFirstDraftPolicy();
	pending->spec_active = 1u;
	pending->spec_first_draft_miss = 0u;
	pending->spec_fold = 0u;
	memset(pending->spec,0,sizeof(pending->spec));
	status = SPARK_STATUS_OK;
	for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
	{
		SparkQwen38_27bServingSpecState *spec;
		uint32_t slot,last_row;
		uint64_t position,sequence;
		uint32_t token;
		uint32_t fold_active;
		fold_active = 0u;
		spec = &pending->spec[lane];
		slot = submission->lanes[lane].resident_sequence_slot;
		last_row = pending->last_row_by_lane[lane];
		position = submission->row_positions[last_row];
		sequence = submission->row_sequence_ids[last_row];
		token = submission->token_ids[last_row];
		spec->resident_slot = slot;
		spec->base_position = position + 1u;
		spec->sequence_id = sequence;
		spec->snapshot_index = lane;
		spec->draft_token_count = draft_count;
		if ( SparkQwen38_27bServingBlockDraftMethod(spec_method) )
		{
			const char *fold_env = getenv("SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD");
			uint32_t fold_mode = fold_env != 0 && fold_env[0] != '0' ? (uint32_t)strtoul(fold_env,0,0) : 0u;
			if ( fold_mode == 0u )
				fold_mode = fold_env != 0 && fold_env[0] != '0' ? 1u : 0u;
			if ( fold_mode >= 1u && state->dflash2_fold_armed != 0u
				&& state->dflash2_fold_sequence_id == sequence
				&& state->dflash2_fold_position == position )
			{
				fold_active = fold_mode >= 2u ? 2u : 1u;
				spec->base_position = position;
				spec->draft_ids[0] = 0u;
				for (draft=1u; draft<draft_count; draft++)
					spec->draft_ids[draft] = state->dflash2_next_draft_ids[draft - 1u];
			}
			else
			{
				state->dflash2_fold_armed = 0u;
				memset(&dspark_draft,0,sizeof(dspark_draft));
				dspark_draft.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
				dspark_draft.descriptor_bytes = sizeof(dspark_draft);
				dspark_draft.block_size = draft_count;
				dspark_draft.draft_token_count = spec_method == SPARK_QWEN38_27B_SERVING_SPEC_METHOD_DFLASH2 ? draft_count - 1u : draft_count;
				dspark_draft.sequence_id = sequence;
				dspark_draft.base_position = spec->base_position;
				dspark_draft.tap_buffer = 0;
				dspark_draft.draft_token_ids = state->dflash2_next_draft_ids;
				status = SparkQwen38_27bServingRunSpeculativeFrame(state,submission,pending,slot,0u,&token,&position,&sequence,1u,0u,submission->sequence_id,submission->sequence_position,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER,0,&dspark_draft,0,1u);
			}
		}
		else
		{
			memset(&mtp_draft,0,sizeof(mtp_draft));
			mtp_draft.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION;
			mtp_draft.descriptor_bytes = sizeof(mtp_draft);
			mtp_draft.lane_index = slot;
			mtp_draft.draft_token_count = draft_count;
			mtp_draft.base_position = spec->base_position;
			mtp_draft.sequence_id = sequence;
			mtp_draft.row_token_ids = pending->frame_token_ids;
			status = SparkQwen38_27bServingRunSpeculativeFrame(state,submission,pending,slot,0u,&token,&position,&sequence,1u,0u,submission->sequence_id,submission->sequence_position,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER,&mtp_draft,0,0,1u + draft_count);
		}
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "qwen38_27b_spec_diag decode_frame_failed lane=%u status=%d\n", lane, (int)status);
		if ( status == SPARK_STATUS_OK && fold_active == 0u )
		{
			spec->committed_ids[0] = pending->frame_output_ids[0];
			if ( SparkQwen38_27bServingBlockDraftMethod(spec_method) )
			{
				spec->draft_ids[0] = spec->committed_ids[0];
				for (draft=1u; draft<draft_count; draft++)
					spec->draft_ids[draft] = state->dflash2_next_draft_ids[draft - 1u];
			}
			else
			{
				for (draft=0u; draft<draft_count; draft++)
					spec->draft_ids[draft] = pending->frame_output_ids[1u + draft];
			}
			spec->first_draft_miss = spec->draft_ids[0] != spec->committed_ids[0] ? 1u : 0u;
			spec->chain_dead = (spec->first_draft_miss != 0u && first_draft_policy == SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT) ? 1u : 0u;
			if ( spec->first_draft_miss != 0u )
				fprintf(stderr, "qwen38_27b_spec first_draft_miss lane=%u C0=%u draft0=%u policy=%s\n", lane, spec->committed_ids[0], spec->draft_ids[0], first_draft_policy == SPARK_QWEN38_27B_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT ? "strict" : "recover");
		}
		if ( status == SPARK_STATUS_OK && spec->chain_dead == 0u )
		{
			status = SparkSpeculationSeamStageLocalDraft(state->speculation_seam,submission->request_id,spec->sequence_id,spec->base_position,spec->draft_ids + 1u,draft_count - 1u);
			if ( status == SPARK_STATUS_OK )
				spec->engine_staged = 1u;
			else
				fprintf(stderr, "qwen38_27b_spec_diag stage_local_draft_failed lane=%u status=%d\n", lane, (int)status);
		}
		if ( status == SPARK_STATUS_OK && spec->chain_dead == 0u )
		{
			memset(&gdn_snapshot,0,sizeof(gdn_snapshot));
			gdn_snapshot.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
			gdn_snapshot.descriptor_bytes = sizeof(gdn_snapshot);
			gdn_snapshot.snapshot_index = spec->snapshot_index;
			verify_tokens[0] = fold_active != 0u ? token : spec->committed_ids[0];
			for (draft=1u; draft<draft_count; draft++)
				verify_tokens[draft] = spec->draft_ids[draft];
			{
				uint32_t verify_flags = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY;
				SparkQwen38_27bDsparkDraftView *verify_draft = 0;
				if ( fold_active == 2u )
				{
					verify_flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
					if ( getenv("SPARK_QWEN38_27B_DFLASH2_OF_NORESTORE") != 0 )
						state->dflash2_fold_restore_slot = -1;
					if ( state->dflash2_fold_restore_slot >= 0 )
					{
						verify_flags |= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW;
						gdn_snapshot.snapshot_index = (uint32_t)state->dflash2_fold_restore_slot;
					}
					memset(&dspark_draft,0,sizeof(dspark_draft));
					dspark_draft.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
					dspark_draft.descriptor_bytes = sizeof(dspark_draft);
					dspark_draft.block_size = draft_count;
					dspark_draft.draft_token_count = draft_count - 1u;
					dspark_draft.sequence_id = sequence;
					dspark_draft.base_position = position + 1u;
					dspark_draft.tap_buffer = 0;
					dspark_draft.draft_token_ids = state->dflash2_draft_matrix;
					dspark_draft.multi_block_count = draft_count;
					verify_draft = &dspark_draft;
				}
				status = SparkQwen38_27bServingRunSpeculativeFrame(state,submission,pending,slot,1u,verify_tokens,0,0,draft_count,spec->base_position,sequence,spec->base_position,verify_flags,0,verify_draft,&gdn_snapshot,draft_count);
			}
			if ( status != SPARK_STATUS_OK )
				fprintf(stderr, "qwen38_27b_spec_diag verify_frame_failed lane=%u status=%d\n", lane, (int)status);
		}
		if ( status == SPARK_STATUS_OK )
		{
			for (draft=0u; draft<draft_count; draft++)
				spec->emitted_ids[draft] = pending->frame_output_ids[draft];
			if ( fold_active != 0u )
			{
				spec->committed_ids[0] = spec->emitted_ids[0];
				spec->draft_ids[0] = spec->emitted_ids[0];
				spec->first_draft_miss = 0u;
				spec->chain_dead = 0u;
			}
			if ( spec->chain_dead == 0u )
			{
				status = SparkSpeculationSeamAcceptChain(state->speculation_seam,spec->sequence_id,spec->emitted_ids,draft_count,&verify_result);
				if ( status == SPARK_STATUS_OK )
				{
					spec->accepted_count = verify_result.accepted_draft_token_count;
					spec->engine_staged = 0u;
				}
				else
					fprintf(stderr, "qwen38_27b_spec_diag accept_chain_failed lane=%u status=%d\n", lane, (int)status);
			}
			else
			{
				spec->accepted_count = 0u;
				while ( spec->accepted_count + 1u < draft_count && spec->emitted_ids[spec->accepted_count] == spec->draft_ids[spec->accepted_count + 1u] )
					spec->accepted_count++;
			}
			if ( fold_active != 0u )
				pending->spec_fold = fold_active;
			fprintf(stderr, "qwen38_27b_spec_diag t=%.6f C0=%u accepted=%u drafts=[%u,%u,%u,%u,%u,%u,%u,%u] emitted=[%u,%u,%u,%u,%u,%u,%u,%u]\n",
				(double)clock_gettime_mono_ns() * 1e-9,
				spec->committed_ids[0], spec->accepted_count,
				spec->draft_ids[0], spec->draft_ids[1], spec->draft_ids[2], spec->draft_ids[3],
				spec->draft_ids[4], spec->draft_ids[5], spec->draft_ids[6], spec->draft_ids[7],
				spec->emitted_ids[0], spec->emitted_ids[1], spec->emitted_ids[2], spec->emitted_ids[3],
				spec->emitted_ids[4], spec->emitted_ids[5], spec->emitted_ids[6], spec->emitted_ids[7]);
		}
	}
	min_accepted = 0u;
	pending->spec_chain_dead = 0u;
	if ( status == SPARK_STATUS_OK )
	{
		min_accepted = draft_count - 1u;
		for (lane=0u; lane<submission->active_sequence_count; lane++)
		{
			if ( pending->spec[lane].chain_dead != 0u )
				pending->spec_chain_dead = 1u;
			if ( pending->spec[lane].first_draft_miss != 0u )
				pending->spec_first_draft_miss++;
			if ( pending->spec[lane].accepted_count < min_accepted )
				min_accepted = pending->spec[lane].accepted_count;
		}
		if ( pending->spec_chain_dead != 0u )
			min_accepted = 0u;
		{
			uint32_t commit_overhead = pending->spec_fold == 2u ? 1u : (pending->spec_fold == 1u ? 2u : 3u);
			if ( min_accepted + commit_overhead > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE )
				min_accepted = SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE - commit_overhead;
			pending->spec_tokens_per_sequence = pending->spec_chain_dead != 0u ? 1u : min_accepted + commit_overhead;
		}
		pending->spec_total_accepted = pending->spec_chain_dead != 0u ? 0u : min_accepted * submission->active_sequence_count;
	}
	for (lane=0u; status == SPARK_STATUS_OK && pending->spec_chain_dead == 0u && lane<submission->active_sequence_count; lane++)
	{
		SparkQwen38_27bServingSpecState *spec;
		uint32_t slot;
		uint32_t replay_rows;
		if ( pending->spec_fold == 2u )
		{
			spec = &pending->spec[lane];
			slot = spec->resident_slot;
			(void)slot;
			{
				uint32_t ci = 1u;
				uint32_t d2;
				for (d2 = 0u; d2 < SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE - 1u; d2++)
					state->dflash2_next_draft_ids[d2] = state->dflash2_draft_matrix[min_accepted * (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE - 1u) + d2];
				for (draft=2u; draft<=min_accepted; draft++)
					spec->committed_ids[ci++] = spec->draft_ids[draft];
				spec->committed_ids[ci++] = spec->emitted_ids[min_accepted];
				state->dflash2_fold_armed = 1u;
				state->dflash2_fold_position = spec->base_position + min_accepted + 1u;
				state->dflash2_fold_sequence_id = spec->sequence_id;
				state->dflash2_fold_restore_slot = (int32_t)min_accepted;
			}
			continue;
		}
		uint32_t replay_tokens[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS + 1u];
		uint64_t replay_base;
		spec = &pending->spec[lane];
		slot = spec->resident_slot;
		{
			const char *sel_env = getenv("SPARK_QWEN38_27B_DFLASH2_STATE_SELECT");
			if ( sel_env == 0 || sel_env[0] == '0' )
			{
				uint32_t d2;
				replay_rows = min_accepted + 2u;
				replay_tokens[0] = spec->committed_ids[0];
				for (d2 = 1u; d2 <= min_accepted; d2++)
					replay_tokens[d2] = spec->draft_ids[d2];
				replay_tokens[min_accepted + 1u] = spec->emitted_ids[min_accepted];
				replay_base = spec->base_position;
				gdn_snapshot.snapshot_index = spec->snapshot_index;
				status = SparkQwen38_27bServingRunSpeculativeFrame(state,submission,pending,slot,1u,replay_tokens,0,0,replay_rows,replay_base,spec->sequence_id,replay_base,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST,0,0,&gdn_snapshot,1u);
				goto replay_done;
			}
		}
		replay_rows = 1u;
		replay_tokens[0] = spec->emitted_ids[min_accepted];
		replay_base = spec->base_position + min_accepted + 1u;
		memset(&gdn_snapshot,0,sizeof(gdn_snapshot));
		gdn_snapshot.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
		gdn_snapshot.descriptor_bytes = sizeof(gdn_snapshot);
		gdn_snapshot.snapshot_index = min_accepted >= 7u ? 7u : min_accepted;
		{
			const char *fold_env = getenv("SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD");
			SparkQwen38_27bDsparkDraftView *replay_draft = 0;
			if ( fold_env != 0 && fold_env[0] != '0' )
			{
				memset(&dspark_draft,0,sizeof(dspark_draft));
				dspark_draft.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
				dspark_draft.descriptor_bytes = sizeof(dspark_draft);
				dspark_draft.block_size = draft_count;
				dspark_draft.draft_token_count = draft_count - 1u;
				dspark_draft.sequence_id = spec->sequence_id;
				dspark_draft.base_position = replay_base + 1u;
				dspark_draft.tap_buffer = 0;
				dspark_draft.draft_token_ids = state->dflash2_next_draft_ids;
				replay_draft = &dspark_draft;
			}
			status = SparkQwen38_27bServingRunSpeculativeFrame(state,submission,pending,slot,1u,replay_tokens,0,0,replay_rows,replay_base,spec->sequence_id,replay_base,
				(uint32_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST
				| (uint32_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW
				| (replay_draft != 0 ? (uint32_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER : 0u),
				0,replay_draft,&gdn_snapshot,1u);
			if ( status == SPARK_STATUS_OK && replay_draft != 0 )
			{
				state->dflash2_fold_armed = 1u;
				state->dflash2_fold_position = replay_base + 1u;
				state->dflash2_fold_sequence_id = spec->sequence_id;
				state->dflash2_fold_restore_slot = -1;
			}
		}
		replay_done:;
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "qwen38_27b_spec_diag replay_frame_failed lane=%u status=%d\n", lane, (int)status);
		if ( status == SPARK_STATUS_OK )
		{
			if ( pending->spec_fold != 0u )
			{
				uint32_t commit_index = 1u;
				for (draft=2u; draft<=min_accepted; draft++)
					spec->committed_ids[commit_index++] = spec->draft_ids[draft];
				if ( min_accepted >= 1u )
					spec->committed_ids[commit_index++] = spec->emitted_ids[min_accepted];
				spec->committed_ids[commit_index++] = pending->frame_output_ids[0];
			}
			else
			{
				for (draft=0u; draft<min_accepted; draft++)
					spec->committed_ids[1u + draft] = spec->draft_ids[1u + draft];
				spec->committed_ids[1u + min_accepted] = spec->emitted_ids[min_accepted];
				spec->committed_ids[2u + min_accepted] = pending->frame_output_ids[0];
			}
		}
	}
	if ( status != SPARK_STATUS_OK )
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( pending->spec[lane].engine_staged != 0u )
				(void)SparkSpeculationSeamCancelSequence(state->speculation_seam,pending->spec[lane].sequence_id);
	return(status);
}

static void SparkQwen38_27bServingComplete(
	SparkQwen38_27bServingState *state,
	SparkQwen38_27bServingPending *pending,
	SparkStatus status)
{
	SparkModelServingCompletion completion;
	uint32_t index;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = pending->common.submission_id;
	completion.request_id = pending->common.request_id;
	completion.sequence_id = pending->common.sequence_id;
	completion.sequence_position = pending->common.sequence_position;
	completion.control_generation = pending->common.control_generation;
	completion.transaction_id = pending->common.transaction_id;
	completion.dispatch_generation = pending->common.dispatch_generation;
	completion.request_generation = pending->common.request_generation;
	completion.step_generation = pending->common.step_generation;
	completion.residency = pending->residency;
	completion.accepted_token_count = (uint32_t)(pending->accepted_token_count > UINT32_MAX ? UINT32_MAX : pending->accepted_token_count);
	completion.queue_delay_ns = pending->queue_delay_ns;
	completion.service_time_ns = pending->service_time_ns;
	if ( SparkQwen38_27bServingOwnsFinalHead(state) != 0u && status == SPARK_STATUS_OK && pending->common.active_sequence_count != 0u )
	{
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		if ( pending->spec_active != 0u )
		{
			uint32_t lane,step;
			completion.tokens_per_sequence = pending->spec_tokens_per_sequence;
			completion.token_count = pending->common.active_sequence_count * pending->spec_tokens_per_sequence;
			for (lane=0u; lane<pending->common.active_sequence_count; lane++)
				for (step=0u; step<completion.tokens_per_sequence; step++)
					completion.token_ids[(lane * completion.tokens_per_sequence) + step] = pending->spec[lane].committed_ids[step];
		}
		else
		{
			completion.tokens_per_sequence = 1u;
			completion.token_count = pending->common.active_sequence_count;
			for (index=0u; index<completion.token_count; index++)
				completion.token_ids[index] = pending->output_token_ids[index];
		}
	}
	if ( pending->spec_active != 0u )
	{
		SparkQwen38_27bServingSpecTelemetry telemetry;
		memset(&telemetry,0,sizeof(telemetry));
		telemetry.first_draft_miss_count = pending->spec_first_draft_miss;
		telemetry.first_draft_policy = SparkQwen38_27bServingSpecFirstDraftPolicy();
		completion.completion_flags |= SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION;
		completion.model_extension_kind = SPARK_QWEN38_27B_SERVING_EXTENSION_KIND;
		completion.model_extension_bytes = sizeof(telemetry);
		memcpy(completion.model_extension,&telemetry,sizeof(telemetry));
	}
	pending->common.active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static SparkStatus SparkQwen38_27bServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen38_27bServingState *state;
	SparkQwen38_27bServingPending *pending;
	SparkStatus status;
	uint32_t speculate;
	state = (SparkQwen38_27bServingState *)adapter_state;
	status = SparkQwen38_27bServingValidateSubmissionBase(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		status = SparkQwen38_27bServingValidateBoundaries(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "qwen38_27b_submit_reject status=%d kind=%u rows=%u lanes=%u pos=%llu slot=%u\n", (int)status, submission->work_kind, submission->row_count, submission->active_sequence_count, (unsigned long long)(submission->row_count != 0u ? submission->row_positions[0] : 0u), submission->row_count != 0u ? submission->lanes[0].resident_sequence_slot : 0u);
		return(status);
	}
	pending = SparkQwen38_27bServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		fprintf(stderr, "qwen38_27b_debug submission kind=%u rows=%u lanes=%u seqpos=%llu lane0=%u slot0=%u seqid0=%llu pos0=%llu reqgen=%llu subgen=%llu\n",
			(unsigned)submission->work_kind, submission->row_count, submission->active_sequence_count,
			(unsigned long long)submission->sequence_position,
			submission->row_count != 0u ? submission->row_lane_indices[0] : 0u,
			submission->row_count != 0u ? submission->lanes[submission->row_lane_indices[0]].resident_sequence_slot : 0u,
			submission->row_count != 0u ? (unsigned long long)submission->row_sequence_ids[0] : 0u,
			submission->row_count != 0u ? (unsigned long long)submission->row_positions[0] : 0u,
			(unsigned long long)submission->request_generation,
			(unsigned long long)submission->submission_id);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		uint32_t lane;
		state->dflash2_fold_armed = 0u;
		for (lane=0u; lane<submission->active_sequence_count; lane++)
		{
			(void)SparkSpeculationSeamCancelSequence(state->speculation_seam,submission->lanes[lane].sequence_id);
			SparkQwen38_27bServingReleaseLane(state,submission->lanes[lane].resident_sequence_slot);
		}
		pending->common.active_sequence_count = 0u;
		pending->residency = submission->residency;
		SparkQwen38_27bServingComplete(state,pending,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	speculate = 0u;
	status = SparkQwen38_27bServingCoverSubmission(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE && state->speculation_enabled != 0u && SparkQwen38_27bServingOwnsFinalHead(state) != 0u && submission->active_sequence_count == 1u )
	{
		status = SparkQwen38_27bServingExtendSpeculativeCoverage(state,submission);
		if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
		{
			speculate = 0u;
			status = SPARK_STATUS_OK;
		}
		else if ( status == SPARK_STATUS_OK )
			speculate = 1u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bServingUploadBlockTable(state);
	if ( status == SPARK_STATUS_OK && speculate != 0u )
		status = SparkQwen38_27bServingSubmitSpeculativeDecode(state,submission,pending);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
	{
		fprintf(stderr, "qwen38_27b_debug table abi=%u bytes=%u blk_tok=%u lane_count=%u lane_cap=%u stride=%u dev_idx=%p dev_cnt=%p host_idx=%p host_cnt=%p adapter_max_active=%u blocks_per_lane=%u attn_layers=%u\n",
			state->block_table.abi_version, state->block_table.descriptor_bytes,
			state->block_table.block_token_count, state->block_table.lane_count,
			state->block_table.lane_capacity, state->block_table.lane_stride,
			(const void *)state->block_table.physical_block_indices,
			(const void *)state->block_table.lane_physical_block_counts,
			(const void *)state->block_table.host_physical_block_indices,
			(const void *)state->block_table.host_lane_physical_block_counts,
			state->max_active_sequence_count, state->blocks_per_lane,
			state->stage_attn_layer_count);
		state->dflash2_fold_armed = 0u;
		status = SparkQwen38_27bServingRunFrame(state,submission,pending,0u,0u,0u,submission->row_count);
	}
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
	{
		uint32_t lane,wave,chunk_rows;
		state->dflash2_fold_armed = 0u;
		for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
		{
			uint32_t lane_rows;
			lane_rows = 0u;
			for (wave=0u; wave<submission->row_count; wave++)
				lane_rows += submission->row_lane_indices[wave] == lane ? 1u : 0u;
			for (wave=0u; status == SPARK_STATUS_OK && wave<lane_rows; wave+=chunk_rows)
			{
				chunk_rows = lane_rows - wave;
				if ( chunk_rows > state->max_input_row_count )
					chunk_rows = state->max_input_row_count;
				status = SparkQwen38_27bServingRunFrame(state,submission,pending,1u,lane,wave,chunk_rows);
			}
		}
	}
	else if ( status == SPARK_STATUS_OK )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen38_27bServingDropSubmission(state,submission);
		pending->common.active = 0u;
		return(status);
	}
	if ( pending->spec_active != 0u )
	{
		pending->accepted_token_count = pending->spec_total_accepted;
		fprintf(stderr,"qwen38_27b_spec accepted=%u\n",pending->spec_total_accepted);
	}
	SparkQwen38_27bServingCommitSubmission(state,submission,pending);
	SparkQwen38_27bServingComplete(state,pending,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static void SparkQwen38_27bServingDestroy(void *adapter_state)
{
	SparkQwen38_27bServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkQwen38_27bServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkQwen38_27bServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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
	SparkMemoryBufferFree(&state->device_block_indices);
	SparkMemoryBufferFree(&state->device_block_counts);
	SparkMemoryBufferFree(&state->gather_scratch);
	SparkMemoryBufferFree(&state->host_block_indices);
	SparkMemoryBufferFree(&state->block_refs);
	SparkMemoryBufferFree(&state->free_blocks);
	SparkSpeculationSeamDestroy(state->speculation_seam);
	free(state->bridge_host);
	free(state);
}

static SparkStatus SparkQwen38_27bServingAcceptsProgram(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context)
{
	SparkQwen38_27bServingState *state;
	state = (SparkQwen38_27bServingState *)accept_context;
	if ( (program->flags & SPARK_QWEN38_27B_SERVING_REQUIRED_PROGRAM_FLAGS) != SPARK_QWEN38_27B_SERVING_REQUIRED_PROGRAM_FLAGS || program->max_inflight < state->pipeline_slot_count || program->profile == 0 || program->profile->max_active_slots < state->max_active_sequence_count || program->profile->max_new_tokens < state->max_input_row_count )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingLoadDriver(
	SparkQwen38_27bServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkServingAdapterDriverRequest request;
	const SparkModelDriverProgramDescriptor *program;
	SparkStatus status;
	request.contract.driver_model_id = SPARK_QWEN38_27B_SERVING_DRIVER_MODEL_ID;
	request.contract.driver_model_revision = QWEN38_27B_MODEL_REVISION;
	request.contract.driver_stage_name = SPARK_QWEN38_27B_SERVING_STAGE_NAME;
	request.contract.driver_target = SPARK_QWEN38_27B_SERVING_TARGET;
	request.contract.model_description_sha256 = QWEN38_27B_CONTRACT_SHA256;
	request.node_context = 0;
	request.completion_context = state;
	request.completion_function = SparkQwen38_27bServingOrphanDriverCompletion;
	request.wake_function = SparkQwen38_27bServingDriverWake;
	program = 0;
	status = SparkServingAdapterTemplateLoadDriver(&request,configuration,
		&state->driver,&program,SparkQwen38_27bServingAcceptsProgram,state,
		&state->driver_instance);
	state->program = program;
	return(status);
}

static SparkStatus SparkQwen38_27bServingAllocatePools(
	SparkQwen38_27bServingState *state)
{
	uint32_t block;
	uint64_t indices;
	SparkStatus status;
	indices = (uint64_t)state->max_active_sequence_count * state->blocks_per_lane;
	status = SparkMemoryBufferAllocate(&state->host_block_indices,
		SPARK_MEMORY_SPACE_HOST_COHERENT,indices * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->block_refs,
			SPARK_MEMORY_SPACE_HOST_COHERENT,(uint64_t)state->kv_block_count * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		memset(state->block_refs.pointer,0,(size_t)state->block_refs.bytes);
	{
		uint32_t li;
		for (li=0u; li<SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; li++)
		{
			state->lane_prefix_entry[li] = 0xFFu;
			state->lane_prefix_blocks[li] = 0u;
			state->lane_publish_armed[li] = 0u;
			state->lane_restore_armed[li] = 0u;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMemoryBufferAllocate(&state->free_blocks,
			SPARK_MEMORY_SPACE_HOST_COHERENT,(uint64_t)state->kv_block_count * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (block=0u; block<state->kv_block_count; block++)
		((uint32_t *)state->free_blocks.pointer)[block] = state->kv_block_count - 1u - block;
	state->free_block_count = state->kv_block_count;
	{
		uint32_t frame_rows = state->max_active_sequence_count > state->max_input_row_count ?
			state->max_active_sequence_count : state->max_input_row_count;
		if ( SparkMemoryBufferAllocate(&state->gather_scratch,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,(uint64_t)frame_rows * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES) != SPARK_STATUS_OK )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( state->stage_attn_layer_count != 0u )
	{
		if ( SparkMemoryBufferAllocate(&state->device_block_indices,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,indices * sizeof(uint32_t)) != SPARK_STATUS_OK ||
			SparkMemoryBufferAllocate(&state->device_block_counts,
			SPARK_MEMORY_SPACE_DEVICE_PRIVATE,(uint64_t)state->max_active_sequence_count * sizeof(uint32_t)) != SPARK_STATUS_OK )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->block_table.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->block_table.lane_count = state->max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices.pointer;
	state->block_table.lane_physical_block_counts = state->device_block_counts.pointer;
	state->block_table.host_physical_block_indices = state->host_block_indices.pointer;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkQwen38_27bServingState *state;
	uint32_t max_sequence_positions;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkQwen38_27bServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkQwen38_27bServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->stage_index = configuration->stage_index;
	state->first_layer_index = SparkQwen38_27bServingFirstLayer(configuration->stage_index);
	state->stage_layer_count = SparkQwen38_27bServingDescriptor.stage_layer_counts[configuration->stage_index];
	state->stage_attn_layer_count = SparkQwen38_27bServingStageAttentionLayers(state->first_layer_index,state->stage_layer_count);
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
	state->shim.execution_stream = configuration->execution_stream;
	status = SparkQwen38_27bServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_QWEN38_27B_SERVING_MAX_SEQUENCE_POSITIONS_CAP) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->max_sequence_positions = max_sequence_positions;
		state->blocks_per_lane = (max_sequence_positions + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		state->kv_block_count = state->resident_sequence_capacity * state->blocks_per_lane;
		status = SparkQwen38_27bServingAllocatePools(state);
		state->shim.input_scratch = state->gather_scratch.pointer;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bServingInitializeSpeculationSeam(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bServingSetEnvironment(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bServingLoadDriver(state,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen38_27bServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingPrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count)
{
	(void)adapter_state;
	(void)submissions;
	(void)submission_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bServingResolvePrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	(void)adapter_state;
	(void)submission;
	(void)resolution;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface SparkQwen38_27bServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkQwen38_27bServingDescriptor,
	.initialize = SparkQwen38_27bServingInitialize,
	.destroy = SparkQwen38_27bServingDestroy,
	.validate_submission = SparkQwen38_27bServingValidateSubmission,
	.submit = SparkQwen38_27bServingSubmit,
	.progress = SparkQwen38_27bServingProgress,
	.quiesce = SparkQwen38_27bServingQuiesce,
	.snapshot = SparkQwen38_27bServingSnapshot,
	.prefetch = SparkQwen38_27bServingPrefetch,
	.resolve_prefetch = SparkQwen38_27bServingResolvePrefetch
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkQwen38_27bServingInterface);
}
