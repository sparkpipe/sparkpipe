/*
 * Qwen 3.6 27B serving adapter: the SparkModelServingAdapterInterface face of
 * the qwen38_resident_decode_stage firmware driver.
 *
 * Two structural differences from the glm52/dsv4 adapters, both owned by the
 * module contract in spark_qwen38_resident_decode_stage_firmware.h:
 *
 * - The module is configured through the strict process environment (the
 *   firmware description's runtime_contract lists every variable), not
 *   through a node context struct. The adapter derives the whole slice
 *   environment from its own configuration - stage pack path, PP13 stage
 *   geometry, runtime limits, and the KV pool size implied by the
 *   max_sequence_positions cap - and sets it before driver create. One
 *   resident process hosts one stage, so the process-wide setenv is the
 *   intended channel. SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION is set to 1:
 *   the published recipe this adapter loads is the qualified execution path.
 *
 * - The module's frame contract takes first-class hidden transport callbacks
 *   and a caller-owned paged KV block table with device and host mirrors.
 *   The adapter supplies both: a per-frame transport shim that lands the
 *   submission's hidden boundary in the module's expected contiguity (decode
 *   rows are already contiguous; a multi-lane prefill is round-major across
 *   lanes, so each lane frame's rows are gathered by explicit flat row index
 *   and the frame's output is scattered back the same way), and a block
 *   allocator over the module's KV pool with the host mirror the module
 *   proves coverage against before every launch.
 *
 * Prefill frames are one lane per frame capped at max_active_sequence_count
 * positions, so a multi-lane or over-cap prefill submission is split into a
 * sequence of frames inside submit; execution is submit_return synchronous,
 * and the single serving completion fires after the final frame lands.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen38_model.h"
#include "sparkpipe/spark_qwen38_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen38_serving_adapter.h"
/* Port 2 of 4: the free-stack block allocator is replaced by the paged
 * KV half over the GENERAL prefix-cache core (SparkPrefixCacheCore*),
 * exactly per .agents/pccore-dev/port23_seam_decisions.md Part A. The
 * paged cache owns allocation, content-addressed publishing, LCP
 * matching at admit, refcounted sharing, LRU eviction, and release;
 * this adapter keeps only geometry and the upload of the resulting
 * lane table rows (its own cudaMemcpyAsync wrappers - unchanged). */
#include "spark_qwen38_paged_kv.h"
#include "runtime/adapter_common.h"

#ifndef QWEN38_MODEL_REVISION
#error "QWEN38_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef QWEN38_CONTRACT_SHA256
#error "QWEN38_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_QWEN38_SERVING_ADAPTER_ID \
	"spark.qwen38.serving-adapter.tp4-pp4.v1"
#define SPARK_QWEN38_SERVING_MODEL_ID "Qwen/Qwen3.8-2.4T-A95B"
#define SPARK_QWEN38_SERVING_DRIVER_MODEL_ID \
	"qwen38.2.4t-a95b.resident-decode-stage-firmware"
#define SPARK_QWEN38_SERVING_STAGE_NAME "qwen38_resident_decode_stage"
#define SPARK_QWEN38_SERVING_TARGET \
	"cuda.sm121.qwen38.resident_decode_stage.fp8"
#define SPARK_QWEN38_SERVING_PROGRAM_NAME "resident_decode"
/* 16 world ranks; stage_index is the world rank. The TP degree is a
 * RUNTIME value from the stage configuration ("tp_degree": 4 -> TP4xPP4,
 * 16 -> TP16xPP1), so one adapter serves both topologies. Pipeline
 * boundaries (hidden in/out, token in/out) are PP boundaries and the
 * per-PP-stage layer counts derive from the degree. The TP-rank tensor
 * sharding (column/row-parallel hidden slices and the router/expert
 * all-reduces) is OUTSTANDING and lands with the rank-local packs. */
#define SPARK_QWEN38_SERVING_STAGE_COUNT 16u
#define SPARK_QWEN38_SERVING_DEFAULT_TP_DEGREE 4u
#define SPARK_QWEN38_SERVING_MAX_PP_STAGE_COUNT 4u
/* Descriptor face: the qualified TP16xPP1 deployment - ONE hybrid
 * TP group covering all 92 layers, so the descriptor validates (per-
 * stage counts were zero and failed SparkModelServingAdapterValidate
 * Descriptor). Other topologies derive their per-stage geometry at
 * initialize from the configuration's tp_degree. */
#define SPARK_QWEN38_SERVING_HYBRID_GROUP_SIZE 16u
#define SPARK_QWEN38_SERVING_DESCRIPTOR_STAGE_LAYER_COUNTS \
	{92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u,92u}
/* Serving caps context at the model's native 262144 until the KV-tier
 * plan lands; the module's KV pool is sized from the deployment's
 * kv_block_count, and the cap merely refuses configs past the model. */
#define SPARK_QWEN38_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_QWEN38_MODEL_MAXIMUM_CONTEXT_TOKENS
/* P0 (audit section 1.6): the module REFUSES max_active_sequences > 409 -
 * LmRouteBuild prices 32-row tiles past 409 rows while the grouped scalar
 * expert kernel walks 16-row tiles, so a wider batch silently skips rows
 * in every expert group. Until the MoE moves to the launch-planner GEMM,
 * this constant is the ONE authoritative active-width ceiling: the
 * descriptor face, the forwarded stage environment and every per-submission
 * width check read it, so a deployment asking for 512 gets a working 400
 * instead of a dead initialize (config_batch_too_wide capacity_exceeded). */
#define SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE 400u
#define SPARK_QWEN38_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

static const char *const SparkQwen38ServingConfigurationMembers[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"tp_degree"
};

/* Prefix caching (paged KV over SparkPrefixCacheCore). On by default;
 * "0" runs the same frames over a reuse-disabled ledger - that switch
 * IS the A/B byte-identity gate, so it must stay a pure toggle: the
 * walked rows, their tokens and every emitted id are identical either
 * way, only the block LEDGER behind the lanes differs. */
#define SPARK_QWEN38_SERVING_PREFIX_CACHE_ENV "SPARK_QWEN38_SERVING_PREFIX_CACHE"

/* Default-ON exact-zero-off semantics live in SparkAdapterEnvFlagDefaultOn
 * (adapter_common.h); the env NAME stays family (frozen taxonomy). */
static uint32_t SparkQwen38ServingPrefixCacheEnabled(void)
{
	return(SparkAdapterEnvFlagDefaultOn(SPARK_QWEN38_SERVING_PREFIX_CACHE_ENV));
}

typedef struct SparkQwen38ServingPending
{
	/* Shape + nine-field submission identity echo: captured once at claim
	 * by SparkAdapterPendingCapture (adapter_common.h). */
	SparkAdapterPendingCore core;
	struct SparkQwen38ServingState *owner;
	/* The frame currently inside the driver; completion matches against it. */
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkQwen38ServingPending;

/* Per-frame transport shim state. The module calls post_receive/send through
 * the frame context; the shim moves the submission boundary into the frame's
 * expected contiguity. Decode rows are contiguous. Prefill frames are one
 * lane each while the submission boundary is round-major across lanes, so a
 * lane's rows sit at irregular flat offsets whenever lane lengths differ;
 * the row maps give each frame row's flat index in the submission buffer
 * (NULL means the frame rows are contiguous from the base). */
typedef struct SparkQwen38ServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkQwen38ServingTransportShim;

typedef struct SparkQwen38ServingState
{
	/* Counters, routes, the loaded-driver handle, residency inputs, the
	 * pending table pointer, and the quiescing latch: the prologue every
	 * family duplicated now lives in SparkAdapterCommonState.
	 * (adapter_common.h). */
	SparkAdapterCommonState common;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
	uint32_t tp_degree;
	uint32_t pp_stage_count;
	uint32_t stage_layer_counts[SPARK_QWEN38_SERVING_MAX_PP_STAGE_COUNT];
	uint32_t first_layer_index;
	uint32_t stage_layer_count;
	uint32_t stage_attn_layer_count;
	uint32_t max_sequence_positions;
	uint32_t blocks_per_lane;
	uint32_t kv_block_count;
	SparkQwen38KvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	/* Port 2: the LIFO free stack (free_blocks/free_block_count) is
	 * GONE. The paged cache over SparkPrefixCacheCore is the single
	 * block ledger: allocation, publish, match, share, evict, release.
	 * reuse_enabled == 0 (env off or no checkpoint slots) keeps the same
	 * ledger with matching disarmed - never a second mechanism. */
	SparkQwen38PagedKv paged;
	uint32_t paged_ready;
	uint32_t prefix_enabled;
	/* One lane's row ids gathered for admit/append (sized to the row
	 * budget - a B512 deployment with long prompts collects 512 ids). */
	uint32_t *match_tokens;
	uint32_t lane_block_counts[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkQwen38ServingTransportShim shim;
	SparkQwen38ServingPending pending[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkQwen38ServingState;

static uint32_t SparkQwen38ServingPpStageIndex(const SparkQwen38ServingState *state, uint32_t world_rank)
{
	return(state->tp_degree != 0u ? world_rank / state->tp_degree : world_rank / SPARK_QWEN38_SERVING_DEFAULT_TP_DEGREE);
}

static uint32_t SparkQwen38ServingPpStageCount(const SparkQwen38ServingState *state)
{
	return(state->pp_stage_count != 0u ? state->pp_stage_count : SPARK_QWEN38_SERVING_STAGE_COUNT / SPARK_QWEN38_SERVING_DEFAULT_TP_DEGREE);
}

static const SparkModelServingAdapterDescriptor SparkQwen38ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP,
	.stage_count = SPARK_QWEN38_SERVING_STAGE_COUNT,
	.parallel_group_size = SPARK_QWEN38_SERVING_HYBRID_GROUP_SIZE,
	.layer_count = SPARK_QWEN38_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	/* The lean module executes every frame on ONE slot (slots[0]) with a
	 * per-frame stream sync, so concurrent inflight frames would share one
	 * set of buffers; advertise 1 until multi-slot pipelining lands. */
	.max_inflight_submission_count = 1u,
	/* Active/input widths advertise the EFFECTIVE clamp, not the firmware
	 * constant: the module refuses batches wider than 409 (audit 1.6), so a
	 * truthful descriptor must never promise 512. Residency/output stay at
	 * the firmware constant - they are pool sizes, not per-frame rows. */
	.max_active_sequence_count = SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE,
	.max_input_row_count = SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE,
	.max_resident_sequence_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_QWEN38_SERVING_ADAPTER_ID,
	.model_id = SPARK_QWEN38_SERVING_MODEL_ID,
	.model_revision = QWEN38_MODEL_REVISION,
	.driver_program_name = SPARK_QWEN38_SERVING_PROGRAM_NAME,
	.artifact_sha256 = QWEN38_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_QWEN38_SERVING_DESCRIPTOR_STAGE_LAYER_COUNTS,
	.minimum_efficient_submission_row_count = 0u
};

static SparkStatus SparkQwen38ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkQwen38ServingState *state,
	uint32_t *max_sequence_positions)
{
	SparkJsonDocument document;
	int32_t root,token;
	uint32_t schema_version;
	char *relative_stage_pack_path;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkQwen38ServingConfigurationMembers,(uint32_t)(sizeof(SparkQwen38ServingConfigurationMembers) / sizeof(SparkQwen38ServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_QWEN38_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,QWEN38_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"tp_degree",&state->tp_degree);
	if ( status == SPARK_STATUS_OK && (state->tp_degree == 0u || SPARK_QWEN38_SERVING_STAGE_COUNT % state->tp_degree != 0u || state->tp_degree > SPARK_QWEN38_SERVING_STAGE_COUNT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"max_sequence_positions",max_sequence_positions);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static uint32_t SparkQwen38ServingFirstLayer(const SparkQwen38ServingState *state, uint32_t stage_index)
{
	uint32_t index,first_layer,pp_stage;
	first_layer = 0u;
	pp_stage = SparkQwen38ServingPpStageIndex(state,stage_index);
	for (index=0u; index<pp_stage; index++)
		first_layer += state->stage_layer_counts[index];
	return(first_layer);
}

static uint32_t SparkQwen38ServingStageAttentionLayers(uint32_t first_layer, uint32_t layer_count)
{
	uint32_t layer,count;
	count = 0u;
	for (layer=first_layer; layer<first_layer+layer_count; layer++)
		count += SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer) == 0u ? 1u : 0u;
	return(count);
}

static SparkStatus SparkQwen38ServingSetEnvironment(
	const SparkQwen38ServingState *state)
{
	char value[32];
#define SPARK_QWEN38_SERVING_SET_TEXT(name,text) \
	do { if ( setenv(name,text,1) != 0 ) return(SPARK_STATUS_INTERNAL_ERROR); } while (0)
#define SPARK_QWEN38_SERVING_SET_UNSIGNED(name,number) \
	do { snprintf(value,sizeof(value),"%u",(uint32_t)(number)); SPARK_QWEN38_SERVING_SET_TEXT(name,value); } while (0)
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION","1");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_PACK_PATH",state->stage_pack_path);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_COUNT",SPARK_QWEN38_SERVING_STAGE_COUNT);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_INDEX",state->stage_index);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_FIRST_LAYER",state->first_layer_index);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_LAYER_COUNT",state->stage_layer_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES",state->common.max_active_sequence_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_PIPELINE_SLOTS",state->common.pipeline_slot_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_KV_BLOCKS",state->kv_block_count);
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_MTP","0");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_GDN_SNAPSHOT_SLOTS","0");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_KV_STORE","none");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_KV_SERVICE","none");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_KV_SOCKET","none");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_KV_POOL_BYTES","0");
	SPARK_QWEN38_SERVING_SET_TEXT("SPARK_QWEN38_STAGE_KV_WORKERS","0");
#undef SPARK_QWEN38_SERVING_SET_TEXT
#undef SPARK_QWEN38_SERVING_SET_UNSIGNED
	return(SPARK_STATUS_OK);
}

/* Wave-major row order, lane bounds, the positions cap, and distinct
 * resident slots. Identical discipline to the glm52 adapter plus the slot
 * uniqueness the qwen38 paged KV table requires: two submission lanes
 * aliasing one resident slot would silently share a KV and GDN state. */
static SparkStatus SparkQwen38ServingValidateRowOrder(
	const SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint8_t slot_seen[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->common.resident_sequence_capacity || slot_seen[slot] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		slot_seen[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
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

/* Hidden boundary pointers exist only after the resident commits a route:
 * the wire submission validate_submission sees always has them absent (the
 * serving header documents this), so this check is meaningful only from
 * submit, never from validate_submission. */
static SparkStatus SparkQwen38ServingValidateBoundaries(
	const SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	uint32_t pp_stage;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES;
	pp_stage = SparkQwen38ServingPpStageIndex(state,state->stage_index);
	if ( (pp_stage != 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (pp_stage == 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (pp_stage + 1u < SparkQwen38ServingPpStageCount(state) && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (pp_stage + 1u == SparkQwen38ServingPpStageCount(state) && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingValidateSubmissionBase(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	/* Presence + quiescing gate + runtime-submission validation (shared). */
	status = SparkAdapterValidateSubmissionOpen(&state->common,
		&SparkQwen38ServingDescriptor,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkQwen38ServingValidateRowOrder(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->model_extension_bytes != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen38ServingState *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SparkQwen38ServingState *)adapter_state;
	status = SparkQwen38ServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}

static SparkQwen38ServingPending *SparkQwen38ServingReservePending(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen38ServingPending *pending;
	int32_t index;
	index = SparkAdapterPendingClaim(state->common.pending,
		sizeof(SparkQwen38ServingPending),state->common.pipeline_slot_count);
	if ( index < 0 )
		return(0);
	pending = &state->pending[index];
	memset(pending,0,sizeof(*pending));
	pending->owner = state;
	/* Early active-stamp preserved: qwen38 stamps active before the fills. */
	pending->core.active = 1u;
	SparkAdapterPendingCapture(&pending->core,submission);
	pending->frame_status = SPARK_STATUS_OK;
	/* The two captures the old loops made separately, now shared helpers. */
	SparkAdapterCaptureLastRowByLane(submission,pending->last_row_by_lane);
	SparkAdapterCaptureResidentSlotsPerLane(submission,
		submission->active_sequence_count,pending->resident_slots);
	return(pending);
}

/* Pre-route driver completions bump the shared orphan counter through
 * SparkAdapterOrphanDriverCompletion and wakes forward through
 * SparkAdapterDispatchWake; both ride the common state at create time.
 */

static void SparkQwen38ServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkQwen38ServingPending *pending;
	SparkQwen38ServingState *state;
	uint32_t matches;
	pending = (SparkQwen38ServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->core.active == 0u || driver_completion == 0 )
		return;
	matches = SparkAdapterDriverCompletionMatches(driver_completion,
		pending->core.identity.request_id,pending->frame_sequence_id,
		pending->frame_sequence_position,state->common.program->program_id);
	if ( matches == 0u )
	{
		/* Mismatch policy stays family: poison the frame status so submit's
		 * synchronous status read fails; the orphan lands in the shared
		 * counter. */
		state->common.orphan_completion_count++;
		pending->frame_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	pending->frame_status = (SparkStatus)driver_completion->status;
	/* Residency mirror school: the LAST matched frame's token parks here and
	 * Complete echoes it - this stays family on purpose. */
	pending->residency = driver_completion->residency;
	SparkAdapterAccumulateFrameCounters(&pending->accepted_token_count,
		&pending->queue_delay_ns,&pending->service_time_ns,driver_completion);
}

/* Lane block bookkeeping (port 2). A lane whose frame range starts at
 * position zero is a (re)start: its core sequence is released - the
 * published blocks STAY CACHED and become LCP-matchable for later
 * admits instead of vanishing into a free stack - and any borrowed
 * scratch returns. On any failure the lane is dropped back to cold so
 * the next touch is a position-zero reset, matching the module's own
 * continuity invalidation. */
static void SparkQwen38ServingReleaseLane(
	SparkQwen38ServingState *state,
	uint32_t slot)
{
	if ( state->paged_ready != 0u )
		SparkQwen38PagedKvLaneReset(&state->paged,slot);
	state->lane_context_tokens[slot] = 0u;
}

/* Collect one lane's row tokens at or above min_position, in row order
 * (rows arrive position-ascending per lane). Returns the count. */
static uint32_t SparkQwen38ServingGatherLaneTokens(
	const SparkModelServingSubmission *submission,
	uint32_t lane,
	uint64_t min_position,
	uint32_t *tokens,
	uint32_t capacity)
{
	uint32_t row,count;
	count = 0u;
	for (row=0u; row<submission->row_count; row++)
	{
		if ( submission->row_lane_indices[row] != lane ||
			submission->row_positions[row] < min_position )
			continue;
		if ( count == capacity )
			break;
		tokens[count++] = submission->token_ids[row];
	}
	return(count);
}

/* Longest published-prefix reuse for one cold lane's prompt: admit over
 * the core, clamp to the deepest live witnessed checkpoint (the GDN
 * law - 69 of 92 layers cannot skip a walk), and return the matched
 * token count (block-granular; 0 = ordinary full walk). Port-2 scope:
 * matched blocks are ADOPTED into the lane's ledger for sharing and
 * accounting; the walked rows are unchanged until the CP-B modifiers
 * land module-side (preflight §4b). */
static uint32_t SparkQwen38ServingMatchPrefix(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission,
	uint32_t lane)
{
	SparkQwen38PagedKvMatch match;
	uint32_t *tokens;
	uint32_t count,slot;
	tokens = state->match_tokens;
	if ( state->prefix_enabled == 0u || state->stage_attn_layer_count == 0u ||
		submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_PREFILL ||
		tokens == 0 )
		return(0u);
	count = SparkQwen38ServingGatherLaneTokens(submission,lane,0,tokens,state->common.max_input_row_count);
	slot = submission->lanes[lane].resident_sequence_slot;
	if ( count == 0u )
		return(0u);
	/* Admit binds the lane's core sequence over the whole prompt: the
	 * core matches its published chains, the witness clamp keeps only
	 * the deepest live witnessed boundary (re-admitting the truncated
	 * shared prefix), and the appended tail publishes new boundaries so
	 * this lane becomes a future donor. */
	if ( SparkQwen38PagedKvAdmit(&state->paged,slot,tokens,count,&match) != SPARK_STATUS_OK || match.block_count == 0u )
		return(0u);
	return(match.block_count * state->paged.configuration.block_token_count);
}

static SparkStatus SparkQwen38ServingCoverLane(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission,
	uint32_t lane,
	uint64_t end_position)
{
	uint64_t committed;
	uint32_t slot,token_count;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	slot = submission->lanes[lane].resident_sequence_slot;
	/* Continuation ids past the sequence's committed frontier enter the
	 * core sequence (canonical publishing); coverage growth beyond it
	 * borrows private scratch. The old free stack answered "is there
	 * room" with a pop AFTER partial lanes were already covered - the
	 * paged cache reserves up front and Trims under pressure, so
	 * exhaustion refuses BEFORE any lane is touched instead of dropping
	 * a submission mid-extend. */
	committed = SparkQwen38PagedKvCommittedTokens(&state->paged,slot);
	token_count = SparkQwen38ServingGatherLaneTokens(submission,lane,committed,state->match_tokens,state->common.max_input_row_count);
	return(SparkQwen38PagedKvCover(&state->paged,slot,end_position,token_count != 0u ? state->match_tokens : 0,token_count));
}

static void SparkQwen38ServingDropSubmission(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		SparkQwen38ServingReleaseLane(state,submission->lanes[lane].resident_sequence_slot);
}

static SparkStatus SparkQwen38ServingCoverSubmission(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	SparkStatus status;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		uint64_t first_position,end_position;
		slot = submission->lanes[lane].resident_sequence_slot;
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
			SparkQwen38ServingReleaseLane(state,slot);
		/* Cold PREFILL lanes are DEFERRED to the frame loop below: the
		 * admit must see the witnesses earlier lanes of the SAME batched
		 * submission bound, or intra-batch prefix sharing never fires.
		 * Warm continuations (first position > 0) are covered here and
		 * must NOT re-admit: Admit owns its lane from zero and would
		 * reset a live residency. Pressure is Trim'd inside the
		 * admit/borrow path, so exhaustion refuses up front - never as a
		 * mid-extend CAPACITY_EXCEEDED. */
		if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL && first_position == 0u )
			continue;
		if ( end_position == 0u )
			continue; /* nothing to cover (e.g. a release-shaped probe) */
		status = SparkQwen38ServingCoverLane(state,submission,lane,end_position);
		if ( status != SPARK_STATUS_OK )
		{
			/* Coverage failure is KV exhaustion: drop every lane the
			 * submission touches so a partial allocation cannot linger.
			 * (The free stack's scratch-lifetime hazard class - the old
			 * comment here documented it - cannot occur: every attached
			 * block is core-ledger owned and LaneReset returns it.) */
			SparkQwen38ServingDropSubmission(state,submission);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen38ServingCommitSubmission(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		for (row=0u; row<submission->row_count; row++)
			if ( submission->row_lane_indices[row] == lane && submission->row_positions[row] + 1u > state->lane_context_tokens[slot] )
				state->lane_context_tokens[slot] = submission->row_positions[row] + 1u;
	}
}

/* Upload only the lanes the submission touched: the first cut moved the
 * whole lane_count x lane_stride table (8 MB at 512 x 4096) every step;
 * a lane slice is 16 KB and only the frame's lanes are read on device. */
static SparkStatus SparkQwen38ServingUploadBlockTable(
	const SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	cudaError_t error = cudaSuccess;
	uint64_t lane_slice_bytes,counts_bytes;
	uint32_t lane,slot;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	lane_slice_bytes = (uint64_t)state->blocks_per_lane * sizeof(uint32_t);
	counts_bytes = (uint64_t)state->common.max_active_sequence_count * sizeof(uint32_t);
	for (lane=0u; error == cudaSuccess && lane<submission->active_sequence_count; lane++)
	{
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->common.max_active_sequence_count )
			continue;
		error = cudaMemcpyAsync((uint8_t *)state->device_block_indices + ((uint64_t)slot * lane_slice_bytes),(const uint8_t *)state->host_block_indices + ((uint64_t)slot * lane_slice_bytes),(size_t)lane_slice_bytes,cudaMemcpyHostToDevice,(cudaStream_t)state->common.execution_stream);
	}
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(state->device_block_counts,state->lane_block_counts,(size_t)counts_bytes,cudaMemcpyHostToDevice,(cudaStream_t)state->common.execution_stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize((cudaStream_t)state->common.execution_stream);
	if ( error != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingPostReceive(
	SparkHiddenTransportSession *transport_session,
	SparkHiddenTransportPacket *packet)
{
	SparkQwen38ServingTransportShim *shim;
	const void *source;
	uint32_t row;
	shim = (SparkQwen38ServingTransportShim *)transport_session;
	if ( shim == 0 || packet == 0 || shim->input_base == 0 || shim->input_rows == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	source = shim->input_base;
	if ( shim->input_row_map != 0 )
	{
		for (row=0u; row<shim->input_rows; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->input_scratch + ((uint64_t)row * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES),(const uint8_t *)shim->input_base + ((uint64_t)shim->input_row_map[row] * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES),SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)shim->execution_stream) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
		if ( cudaStreamSynchronize((cudaStream_t)shim->execution_stream) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
		source = shim->input_scratch;
	}
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = shim->input_rows;
	packet->hidden_dimension = SPARK_QWEN38_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES;
	packet->hidden_bf16 = source;
	packet->cuda_stream = shim->execution_stream;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingSend(
	SparkHiddenTransportSession *transport_session,
	const SparkHiddenTransportPacket *packet)
{
	SparkQwen38ServingTransportShim *shim;
	uint32_t row;
	shim = (SparkQwen38ServingTransportShim *)transport_session;
	if ( shim == 0 || packet == 0 || packet->hidden_bf16 == 0 || shim->output_base == 0 || packet->active_sequence_count == 0u || packet->hidden_dimension != SPARK_QWEN38_MODEL_HIDDEN_DIMENSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( shim->output_row_map != 0 )
	{
		for (row=0u; row<packet->active_sequence_count; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->output_base + ((uint64_t)shim->output_row_map[row] * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES),(const uint8_t *)packet->hidden_bf16 + ((uint64_t)row * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES),SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
	}
	else if ( cudaMemcpyAsync(shim->output_base,packet->hidden_bf16,(uint64_t)packet->active_sequence_count * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	return(cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static void SparkQwen38ServingBuildFrame(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38ServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows,
	SparkQwen38DecodeBatchView *decode_batch,
	SparkQwen38PrefillFrameView *prefill_view,
	SparkQwen38ResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t slot;
	uint64_t base_position;
	uint32_t row;
	slot = prefill != 0u ? pending->resident_slots[lane] : 0u;
	base_position = 0u;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
		context->kv_block_table = &state->block_table;
	}
	if ( SparkQwen38ServingPpStageIndex(state,state->stage_index) != 0u )
	{
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen38ServingPostReceive;
	}
	if ( SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u < SparkQwen38ServingPpStageCount(state) )
	{
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SparkQwen38ServingSend;
	}
	state->shim.input_base = submission->hidden_input_address;
	state->shim.input_rows = frame_rows;
	state->shim.output_base = submission->hidden_output_address;
	if ( prefill != 0u )
	{
		/* Round-major submissions interleave lanes by wave, so with unequal
		 * lane lengths a lane's rows sit at irregular flat offsets; gather
		 * the lane's rows by explicit flat index instead of a fixed pitch. */
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
		prefill_view->abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = submission->row_sequence_ids[pending->frame_row_flats[0]];
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW;
		context->prefill_frame = prefill_view;
	}
	else
	{
		for (row=0u; row<frame_rows; row++)
			pending->frame_row_slots[row] = pending->resident_slots[submission->row_lane_indices[row]];
		memcpy(pending->frame_token_ids,submission->token_ids,(size_t)frame_rows * sizeof(uint32_t));
		state->shim.input_row_map = 0;
		state->shim.output_row_map = 0;
		decode_batch->abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = submission->row_positions;
		decode_batch->row_sequence_ids = submission->row_sequence_ids;
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
		context->decode_batch = decode_batch;
	}
	memset(buffers,0,sizeof(SparkModelDriverBuffer[2]));
	if ( SparkQwen38ServingPpStageIndex(state,state->stage_index) == 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u == SparkQwen38ServingPpStageCount(state) )
	{
		uint32_t out_index;
		out_index = SparkQwen38ServingPpStageIndex(state,state->stage_index) == 0u ? 1u : 0u;
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
	frame->program_id = state->common.program->program_id;
	frame->execution_stream = state->common.execution_stream;
	frame->buffers = SparkQwen38ServingPpStageIndex(state,state->stage_index) == 0u || SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u == SparkQwen38ServingPpStageCount(state) ? buffers : 0;
	frame->buffer_count = (SparkQwen38ServingPpStageIndex(state,state->stage_index) == 0u ? 1u : 0u) + (SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u == SparkQwen38ServingPpStageCount(state) ? 1u : 0u);
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkQwen38ServingDriverCompletion;
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SparkQwen38ServingAdmit(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	(void)submission;
	return(SparkAdapterAdmitFrame(&state->common.driver,
		state->common.driver_instance,state->common.program,0,frame,0u));
}

static SparkStatus SparkQwen38ServingRunFrame(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen38ServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows)
{
	SparkQwen38DecodeBatchView decode_batch;
	SparkQwen38PrefillFrameView prefill_view;
	SparkQwen38ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SparkQwen38ServingBuildFrame(state,submission,pending,prefill,lane,wave_base,frame_rows,&decode_batch,&prefill_view,&context,buffers,&frame);
	status = SparkQwen38ServingAdmit(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->common.program->submit(state->common.driver_instance,&frame);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	if ( status == SPARK_STATUS_OK && SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u == SparkQwen38ServingPpStageCount(state) )
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

static void SparkQwen38ServingComplete(
	SparkQwen38ServingState *state,
	SparkQwen38ServingPending *pending,
	SparkStatus status)
{
	SparkModelServingCompletion completion;
	uint32_t index;
	/* ABI stamp + identity echo from the shared builder; residency echoes the
	 * family's pending mirror, not a fixed submission anchor. */
	SparkAdapterBuildCompletionHeaderWithResidency(&completion,
		&pending->core,&pending->residency);
	completion.status = (uint32_t)status;
	completion.accepted_token_count =
		SparkAdapterClampAcceptedTokenCount(pending->accepted_token_count);
	completion.queue_delay_ns = pending->queue_delay_ns;
	completion.service_time_ns = pending->service_time_ns;
	if ( SparkQwen38ServingPpStageIndex(state,state->stage_index) + 1u == SparkQwen38ServingPpStageCount(state) && status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->core.active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[index];
	}
	pending->core.active = 0u;
	state->common.sink.function(state->common.sink.context,&completion);
}

static SparkStatus SparkQwen38ServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen38ServingState *state;
	SparkQwen38ServingPending *pending;
	SparkStatus status;
	state = (SparkQwen38ServingState *)adapter_state;
	status = SparkQwen38ServingValidateSubmissionBase(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		status = SparkQwen38ServingValidateBoundaries(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkQwen38ServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	status = SparkQwen38ServingCoverSubmission(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ServingUploadBlockTable(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		status = SparkQwen38ServingRunFrame(state,submission,pending,0u,0u,0u,submission->row_count);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
	{
		uint32_t lane,wave,chunk_rows;
		for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
		{
			uint32_t lane_rows,slot;
			uint64_t lane_first_position,lane_end_position;
			slot = submission->lanes[lane].resident_sequence_slot;
			lane_first_position = UINT64_MAX;
			lane_end_position = 0u;
			lane_rows = 0u;
			for (wave=0u; wave<submission->row_count; wave++)
				if ( submission->row_lane_indices[wave] == lane )
				{
					lane_rows++;
					if ( submission->row_positions[wave] < lane_first_position )
						lane_first_position = submission->row_positions[wave];
					if ( submission->row_positions[wave] + 1u > lane_end_position )
						lane_end_position = submission->row_positions[wave] + 1u;
				}
			if ( lane_first_position == 0u )
			{
				/* Deferred cold-lane admit (see CoverSubmission): the
				 * donors ahead of this lane in THIS submission have run
				 * and bound their publish-boundary witnesses. */
				if ( state->lane_context_tokens[slot] != 0u )
					SparkQwen38ServingReleaseLane(state,slot);
				(void)SparkQwen38ServingMatchPrefix(state,submission,lane);
				status = SparkQwen38ServingCoverLane(state,submission,lane,lane_end_position);
				if ( status != SPARK_STATUS_OK )
				{
					/* Coverage failure is KV exhaustion: drop every lane
					 * the submission touches so nothing partial lingers. */
					SparkQwen38ServingDropSubmission(state,submission);
					pending->core.active = 0u;
					return(status);
				}
			}
			for (wave=0u; status == SPARK_STATUS_OK && wave<lane_rows; wave+=chunk_rows)
			{
				chunk_rows = lane_rows - wave;
				if ( chunk_rows > state->common.max_active_sequence_count )
					chunk_rows = state->common.max_active_sequence_count;
				status = SparkQwen38ServingRunFrame(state,submission,pending,1u,lane,wave,chunk_rows);
				if ( status == SPARK_STATUS_OK && state->prefix_enabled != 0u && state->stage_attn_layer_count != 0u )
				{
					/* Every publish boundary this frame walked binds an
					 * adapter-owned checkpoint slot (the qwen36
					 * donor-witness pattern verbatim): the boundary is
					 * recorded under the lane's block at that ordinal,
					 * and only a matcher whose attached block IS that
					 * witness may clamp to it. The GDN recurrence
					 * capture itself rides the CP-B modifier landing
					 * (preflight §4b); the witness bookkeeping lands
					 * now so the ledger is already clamp-correct. */
					uint64_t chunk_end = lane_first_position + wave + chunk_rows;
					uint64_t boundary = lane_first_position + wave;
					uint32_t checkpoint_slot;
					boundary -= boundary % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
					for (boundary += SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
						boundary <= chunk_end;
						boundary += SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)
						if ( SparkQwen38PagedKvCheckpointOffer(&state->paged,slot,boundary,&checkpoint_slot) != 0u )
							SparkQwen38PagedKvCheckpointCommit(&state->paged,slot,checkpoint_slot,boundary);
				}
			}
		}
	}
	else if ( status == SPARK_STATUS_OK )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
	{
		/* A failed submission fires no completion, matching the glm52/dsv4
		 * adapters; every lane it touched drops back to cold so the next
		 * touch is a position-zero reset on both sides of the contract. */
		SparkQwen38ServingDropSubmission(state,submission);
		pending->core.active = 0u;
		return(status);
	}
	SparkQwen38ServingCommitSubmission(state,submission);
	SparkQwen38ServingComplete(state,pending,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

/* Quiesce and snapshot are the shared lifecycle bodies over the common state. */
static SparkStatus SparkQwen38ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkQwen38ServingState *state;
	state = (SparkQwen38ServingState *)adapter_state;
	return(SparkAdapterQuiesce(state != 0 ? &state->common : 0,deadline_time_ns));
}

static SparkStatus SparkQwen38ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkQwen38ServingState *state;
	state = (SparkQwen38ServingState *)adapter_state;
	return(SparkAdapterSnapshot(state != 0 ? &state->common : 0,snapshot));
}

static void SparkQwen38ServingDestroy(void *adapter_state)
{
	SparkQwen38ServingState *state;
	state = (SparkQwen38ServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkAdapterDestroyReady(&state->common) == 0u )
		return;
	SparkAdapterTeardownDriver(&state->common);
	if ( state->device_block_indices != 0 )
		(void)cudaFree(state->device_block_indices);
	if ( state->device_block_counts != 0 )
		(void)cudaFree(state->device_block_counts);
	if ( state->gather_scratch != 0 )
		(void)cudaFree(state->gather_scratch);
	if ( state->paged_ready != 0u )
	{
		/* Conserving teardown: every attached block returns through the
		 * core ledger before the pool itself dies. */
		SparkQwen38PagedKvDestroy(&state->paged);
		state->paged_ready = 0u;
	}
	free(state->host_block_indices);
	free(state->match_tokens);
	free(state);
}

static SparkStatus SparkQwen38ServingLoadDriver(
	SparkQwen38ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	/* Kind 1 = the exact flag-mask school with explicit profile floors; the
	 * target IS pinned on this path. The module reads its slice from the
	 * process environment, so node_context stays NULL. */
	SparkAdapterDriverContract contract;
	contract.model_id = SPARK_QWEN38_SERVING_DRIVER_MODEL_ID;
	contract.model_revision = QWEN38_MODEL_REVISION;
	contract.stage_name = SPARK_QWEN38_SERVING_STAGE_NAME;
	contract.target = SPARK_QWEN38_SERVING_TARGET;
	contract.description_sha256 = QWEN38_CONTRACT_SHA256;
	contract.required_program_flags = SPARK_QWEN38_SERVING_REQUIRED_PROGRAM_FLAGS;
	contract.check_kind = 1u;
	contract.node_context = 0;
	return(SparkAdapterLoadDriver(&state->common,configuration,&contract));
}

/* KV bytes for ONE 64-token block across this stage's attention-layer
 * slice at the rank-local KV-head share - computed from stage geometry,
 * never hardcoded (seam memo A.1): an attention row is K+V x
 * head_dimension bf16 per LOCAL KV head per layer (kernels take
 * rank-local heads), times the block's token count. GDN state pools
 * (the other 69 layers) stay lane-local and NEVER join this pool. */
static uint64_t SparkQwen38ServingBlockStrideBytes(
	const SparkQwen38ServingState *state)
{
	uint32_t tp_degree,local_heads;
	tp_degree = state->tp_degree != 0u ? state->tp_degree : SPARK_QWEN38_SERVING_DEFAULT_TP_DEGREE;
	local_heads = tp_degree <= SPARK_QWEN38_MODEL_ATTN_KV_HEAD_COUNT ?
		SPARK_QWEN38_MODEL_ATTN_KV_HEAD_COUNT / tp_degree : 1u;
	if ( local_heads == 0u )
		local_heads = 1u;
	return((uint64_t)state->stage_attn_layer_count * local_heads *
		SPARK_QWEN38_MODEL_ATTN_HEAD_DIMENSION * 2u *
		SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES *
		SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
}

static SparkStatus SparkQwen38ServingAllocatePools(
	SparkQwen38ServingState *state)
{
	SparkQwen38PagedKvConfiguration paged_configuration;
	uint64_t indices;
	SparkStatus status;
	indices = (uint64_t)state->common.max_active_sequence_count * state->blocks_per_lane;
	state->host_block_indices = (uint32_t *)malloc((size_t)indices * sizeof(uint32_t));
	state->match_tokens = (uint32_t *)malloc((size_t)state->common.max_input_row_count * sizeof(uint32_t));
	if ( state->host_block_indices == 0 || state->match_tokens == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc(&state->gather_scratch,(size_t)((uint64_t)state->common.max_active_sequence_count * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	/* The paged KV cache binds the GENERAL prefix-cache core for block
	 * lifetime, content-addressed publishing and admit matching; the
	 * adapter half keeps only geometry and the checkpoint gating. It
	 * borrows this adapter's table rows as its storage, so the uploaded
	 * table IS the paged view - one source of truth. The free stack is
	 * gone: the core free list is the only allocation ledger. */
	memset(&paged_configuration,0,sizeof(paged_configuration));
	paged_configuration.block_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	/* Lanes ARE resident slots: the table must span the whole slot space
	 * (resident_sequence_capacity), not just the active wave - a slot
	 * beyond the active count is still a legal residency target. */
	paged_configuration.lane_count = state->common.resident_sequence_capacity;
	paged_configuration.blocks_per_lane = state->blocks_per_lane;
	paged_configuration.physical_page_capacity = state->kv_block_count;
	/* The pool IS the index bound: this module has no JIT_KV page-budget
	 * channel (its runtime-limit page capacities are pinned zero), so
	 * logical == physical - every resident block can be indexed. */
	paged_configuration.logical_page_capacity = state->kv_block_count;
	/* Reuse needs somewhere to put a donor recurrence witness: a stage
	 * with no GDN layers cannot bind one, so reuse stays off there.
	 * Adapter-owned slots (the module's own snapshot pool stays at 0). */
	paged_configuration.checkpoint_slot_count =
		state->prefix_enabled != 0u &&
		(state->stage_layer_count - state->stage_attn_layer_count) != 0u ?
		SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS : 0u;
	paged_configuration.block_stride_bytes = SparkQwen38ServingBlockStrideBytes(state);
	status = SparkQwen38PagedKvInitialize(&state->paged,&paged_configuration,
		state->host_block_indices,state->lane_block_counts);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->paged_ready = 1u;
	if ( state->stage_attn_layer_count != 0u )
	{
		if ( cudaMalloc((void **)&state->device_block_indices,(size_t)(indices * sizeof(uint32_t))) != cudaSuccess || cudaMalloc((void **)&state->device_block_counts,(size_t)(state->common.max_active_sequence_count * sizeof(uint32_t))) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->block_table.abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->block_table.lane_count = state->common.max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->common.max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices;
	state->block_table.lane_physical_block_counts = state->device_block_counts;
	state->block_table.host_physical_block_indices = state->host_block_indices;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	return(SparkAdapterValidateConfiguration(&SparkQwen38ServingDescriptor,
		configuration,SPARK_QWEN38_SERVING_PROGRAM_NAME,
		SPARK_QWEN38_SERVING_STAGE_COUNT));
}

static SparkStatus SparkQwen38ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkQwen38ServingState *state;
	uint32_t max_sequence_positions;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkQwen38ServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkQwen38ServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->stage_index = configuration->stage_index;
	state->tp_degree = SPARK_QWEN38_SERVING_DEFAULT_TP_DEGREE;
	(void)SparkAdapterInitializePrologue(&state->common,configuration);
	state->common.pending = &state->pending[0].core;
	state->common.pending_element_bytes = sizeof(state->pending[0]);
	/* P0 clamp: forward at most the EFFECTIVE active width regardless of
	 * what the deployment asks for; a deployment requesting 512 used to
	 * die downstream with config_batch_too_wide capacity_exceeded and now
	 * comes up at 400 with this loud line. The clamp narrows the shared
	 * prologue's active-width copies AFTER initialize so LoadDriver's
	 * exact-mask floors and every forwarded env / pool size see it, while
	 * common.runtime_limits keeps the RAW deployment limits for submission
	 * validation - byte-equal with the pre-skeleton behavior. */
	if ( configuration->runtime_limits.max_active_sequence_count > SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE )
		fprintf(stderr,"%s clamp max_active_sequences %u -> %u (module refuses >409 until launch-planner GEMM lands)\n",SPARK_QWEN38_SERVING_STAGE_NAME,configuration->runtime_limits.max_active_sequence_count,(uint32_t)SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE);
	state->common.max_active_sequence_count = configuration->runtime_limits.max_active_sequence_count < SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE ? configuration->runtime_limits.max_active_sequence_count : (uint32_t)SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE;
	state->common.max_input_row_count = configuration->runtime_limits.max_input_row_count < SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE ? configuration->runtime_limits.max_input_row_count : (uint32_t)SPARK_QWEN38_SERVING_MAX_ACTIVE_SEQUENCES_EFFECTIVE;
	state->prefix_enabled = SparkQwen38ServingPrefixCacheEnabled();
	state->shim.execution_stream = configuration->execution_stream;
	status = SparkQwen38ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_QWEN38_SERVING_MAX_SEQUENCE_POSITIONS_CAP) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		/* PP geometry derives from the config's tp_degree: TP4 -> four
		 * stages of 23 layers, TP16 -> one stage of 92. */
		uint32_t pp,base,extra;
		state->pp_stage_count = SPARK_QWEN38_SERVING_STAGE_COUNT / state->tp_degree;
		if ( state->pp_stage_count > SPARK_QWEN38_SERVING_MAX_PP_STAGE_COUNT )
			state->pp_stage_count = SPARK_QWEN38_SERVING_MAX_PP_STAGE_COUNT;
		base = SPARK_QWEN38_MODEL_LAYER_COUNT / state->pp_stage_count;
		extra = SPARK_QWEN38_MODEL_LAYER_COUNT % state->pp_stage_count;
		for (pp = 0u; pp < state->pp_stage_count; pp++)
			state->stage_layer_counts[pp] = base + (pp < extra ? 1u : 0u);
		state->first_layer_index = SparkQwen38ServingFirstLayer(state,configuration->stage_index);
		state->stage_layer_count = state->stage_layer_counts[SparkQwen38ServingPpStageIndex(state,configuration->stage_index)];
		state->stage_attn_layer_count = SparkQwen38ServingStageAttentionLayers(state->first_layer_index,state->stage_layer_count);
		state->max_sequence_positions = max_sequence_positions;
		state->blocks_per_lane = (max_sequence_positions + SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		state->kv_block_count = state->common.resident_sequence_capacity * state->blocks_per_lane;
		status = SparkQwen38ServingAllocatePools(state);
		state->shim.input_scratch = state->gather_scratch;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ServingSetEnvironment(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ServingLoadDriver(state,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen38ServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface SparkQwen38ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkQwen38ServingDescriptor,
	.initialize = SparkQwen38ServingInitialize,
	.destroy = SparkQwen38ServingDestroy,
	.validate_submission = SparkQwen38ServingValidateSubmission,
	.submit = SparkQwen38ServingSubmit,
	.progress = SparkModelServingAdapterStreamOrderedProgress,
	.quiesce = SparkQwen38ServingQuiesce,
	.snapshot = SparkQwen38ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkQwen38ServingInterface);
}
