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
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen38_model.h"
#include "sparkpipe/spark_qwen38_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen38_serving_adapter.h"

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
/* World ranks in the TP4xPP4 deployment: stage_index is the world rank,
 * pp_stage = stage_index / TP_DEGREE. Pipeline boundaries (hidden in/out,
 * token in/out) are PP boundaries; the descriptor's stage_layer_counts is
 * indexed by PP stage. The TP-rank tensor sharding (column/row-parallel
 * hidden slices and the router/expert all-reduces) is OUTSTANDING and lands
 * with the rank-local packs. */
#define SPARK_QWEN38_SERVING_STAGE_COUNT 16u
#define SPARK_QWEN38_SERVING_TP_DEGREE 4u
#define SPARK_QWEN38_SERVING_PP_STAGE_COUNT \
	(SPARK_QWEN38_SERVING_STAGE_COUNT / SPARK_QWEN38_SERVING_TP_DEGREE)

static uint32_t SparkQwen38ServingPpStageIndex(uint32_t world_rank)
{
	return(world_rank / SPARK_QWEN38_SERVING_TP_DEGREE);
}
/* Serving caps context at the model's native 262144 until the KV-tier
 * plan lands; the module's KV pool is sized from the deployment's
 * kv_block_count, and the cap merely refuses configs past the model. */
#define SPARK_QWEN38_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_QWEN38_MODEL_MAXIMUM_CONTEXT_TOKENS
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
	"max_sequence_positions"
};

typedef struct SparkQwen38ServingPending
{
	struct SparkQwen38ServingState *owner;
	uint32_t active;
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
	uint64_t orphan_completion_count;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkQwen38KvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkQwen38ServingTransportShim shim;
	SparkQwen38ServingPending pending[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkQwen38ServingState;

static const SparkModelServingAdapterDescriptor SparkQwen38ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV,
	.stage_count = SPARK_QWEN38_SERVING_STAGE_COUNT,
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
	.max_active_sequence_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_QWEN38_SERVING_ADAPTER_ID,
	.model_id = SPARK_QWEN38_SERVING_MODEL_ID,
	.model_revision = QWEN38_MODEL_REVISION,
	.driver_program_name = SPARK_QWEN38_SERVING_PROGRAM_NAME,
	.artifact_sha256 = QWEN38_CONTRACT_SHA256,
	.stage_layer_counts = {23u,23u,23u,23u},
	.minimum_efficient_submission_row_count = 0u
};

static int32_t SparkQwen38ServingJsonMember(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,root,name));
}

static SparkStatus SparkQwen38ServingJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkQwen38ServingJsonMember(document,root,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

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
		status = SparkQwen38ServingJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_QWEN38_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkQwen38ServingJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,QWEN38_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkQwen38ServingJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ServingJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static uint32_t SparkQwen38ServingFirstLayer(uint32_t stage_index)
{
	uint32_t index,first_layer,pp_stage;
	first_layer = 0u;
	pp_stage = SparkQwen38ServingPpStageIndex(stage_index);
	for (index=0u; index<pp_stage; index++)
		first_layer += SparkQwen38ServingDescriptor.stage_layer_counts[index];
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
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES",state->max_active_sequence_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED("SPARK_QWEN38_STAGE_PIPELINE_SLOTS",state->pipeline_slot_count);
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
		if ( slot >= state->resident_sequence_capacity || slot_seen[slot] != 0u )
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
	pp_stage = SparkQwen38ServingPpStageIndex(state->stage_index);
	if ( (pp_stage != 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (pp_stage == 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (pp_stage + 1u < SPARK_QWEN38_SERVING_PP_STAGE_COUNT && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (pp_stage + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingValidateSubmissionBase(
	SparkQwen38ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkQwen38ServingDescriptor,&state->runtime_limits,submission);
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
			pending->frame_status = SPARK_STATUS_OK;
			for (row=0u; row<submission->row_count; row++)
			{
				lane = submission->row_lane_indices[row];
				pending->last_row_by_lane[lane] = row;
			}
			for (lane=0u; lane<submission->active_sequence_count; lane++)
				pending->resident_slots[lane] = submission->lanes[lane].resident_sequence_slot;
			return(pending);
		}
	}
	return(0);
}

static void SparkQwen38ServingOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkQwen38ServingState *state;
	(void)driver_completion;
	state = (SparkQwen38ServingState *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SparkQwen38ServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkQwen38ServingPending *pending;
	SparkQwen38ServingState *state;
	uint32_t matches;
	pending = (SparkQwen38ServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->active == 0u || driver_completion == 0 )
		return;
	matches = driver_completion->request_id == pending->request_id && driver_completion->sequence_id == pending->frame_sequence_id && driver_completion->sequence_position == pending->frame_sequence_position && driver_completion->program_id == state->program->program_id;
	if ( matches == 0u )
	{
		state->orphan_completion_count++;
		pending->frame_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	pending->frame_status = (SparkStatus)driver_completion->status;
	pending->residency = driver_completion->residency;
	pending->accepted_token_count += driver_completion->accepted_token_count;
	pending->queue_delay_ns += driver_completion->queue_delay_ns;
	pending->service_time_ns += driver_completion->service_time_ns;
}

static void SparkQwen38ServingDriverWake(void *wake_context)
{
	SparkQwen38ServingState *state;
	state = (SparkQwen38ServingState *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SparkQwen38ServingAvailableSubmissionCount(
	const SparkQwen38ServingState *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].active == 0u ? 1u : 0u;
	return(available);
}

/* Lane block bookkeeping. A lane whose frame range starts at position zero
 * is a (re)start: its old blocks return to the free stack before the new
 * coverage is allocated. On any failure the lane is dropped back to cold so
 * the next touch is a position-zero reset, matching the module's own
 * continuity invalidation. */
static void SparkQwen38ServingReleaseLane(
	SparkQwen38ServingState *state,
	uint32_t slot)
{
	uint32_t ordinal;
	for (ordinal=0u; ordinal<state->lane_block_counts[slot]; ordinal++)
		state->free_blocks[state->free_block_count++] = state->host_block_indices[((uint64_t)slot * state->blocks_per_lane) + ordinal];
	state->lane_block_counts[slot] = 0u;
	state->lane_context_tokens[slot] = 0u;
}

static SparkStatus SparkQwen38ServingCoverLane(
	SparkQwen38ServingState *state,
	uint32_t slot,
	uint64_t end_position)
{
	uint32_t required,ordinal;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	required = (uint32_t)((end_position + SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	if ( required > state->blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (ordinal=state->lane_block_counts[slot]; ordinal<required; ordinal++)
	{
		if ( state->free_block_count == 0u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->host_block_indices[((uint64_t)slot * state->blocks_per_lane) + ordinal] = state->free_blocks[--state->free_block_count];
	}
	state->lane_block_counts[slot] = required;
	return(SPARK_STATUS_OK);
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
		status = SparkQwen38ServingCoverLane(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
		{
			/* Coverage failure is KV exhaustion: drop every lane the
			 * submission touches so a partial allocation cannot linger. */
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

static SparkStatus SparkQwen38ServingUploadBlockTable(
	const SparkQwen38ServingState *state)
{
	cudaError_t error;
	uint64_t indices_bytes,counts_bytes;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	indices_bytes = (uint64_t)state->max_active_sequence_count * state->blocks_per_lane * sizeof(uint32_t);
	counts_bytes = (uint64_t)state->max_active_sequence_count * sizeof(uint32_t);
	error = cudaMemcpy(state->device_block_indices,state->host_block_indices,(size_t)indices_bytes,cudaMemcpyHostToDevice);
	if ( error == cudaSuccess )
		error = cudaMemcpy(state->device_block_counts,state->lane_block_counts,(size_t)counts_bytes,cudaMemcpyHostToDevice);
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
	if ( SparkQwen38ServingPpStageIndex(state->stage_index) != 0u )
	{
		context->flags |= SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen38ServingPostReceive;
	}
	if ( SparkQwen38ServingPpStageIndex(state->stage_index) + 1u < SPARK_QWEN38_SERVING_PP_STAGE_COUNT )
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
	if ( SparkQwen38ServingPpStageIndex(state->stage_index) == 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen38ServingPpStageIndex(state->stage_index) + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT )
	{
		uint32_t out_index;
		out_index = SparkQwen38ServingPpStageIndex(state->stage_index) == 0u ? 1u : 0u;
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
	frame->buffers = SparkQwen38ServingPpStageIndex(state->stage_index) == 0u || SparkQwen38ServingPpStageIndex(state->stage_index) + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT ? buffers : 0;
	frame->buffer_count = (SparkQwen38ServingPpStageIndex(state->stage_index) == 0u ? 1u : 0u) + (SparkQwen38ServingPpStageIndex(state->stage_index) + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT ? 1u : 0u);
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
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = state->program->program_id;
	request.request_id = frame->request_id;
	request.sequence_id = frame->sequence_id;
	request.sequence_position = frame->sequence_position;
	request.deadline_time_ns = frame->deadline_time_ns;
	request.active_slot_count = frame->active_slot_count;
	request.new_token_count = frame->new_token_count;
	request.priority = frame->priority;
	request.frame_flags = frame->flags;
	request.residency = submission->residency;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	status = state->driver.interface->admit(state->driver_instance,&request,&decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( decision.accepted == 0u )
		return(decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY ? SPARK_STATUS_BUSY : SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SparkModelDriverApplyAdmissionDecision(&decision,frame));
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
		status = state->program->submit(state->driver_instance,&frame);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	if ( status == SPARK_STATUS_OK && SparkQwen38ServingPpStageIndex(state->stage_index) + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT )
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
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = pending->submission_id;
	completion.request_id = pending->request_id;
	completion.sequence_id = pending->sequence_id;
	completion.sequence_position = pending->sequence_position;
	completion.control_generation = pending->control_generation;
	completion.transaction_id = pending->transaction_id;
	completion.dispatch_generation = pending->dispatch_generation;
	completion.request_generation = pending->request_generation;
	completion.step_generation = pending->step_generation;
	completion.residency = pending->residency;
	completion.accepted_token_count = (uint32_t)(pending->accepted_token_count > UINT32_MAX ? UINT32_MAX : pending->accepted_token_count);
	completion.queue_delay_ns = pending->queue_delay_ns;
	completion.service_time_ns = pending->service_time_ns;
	if ( SparkQwen38ServingPpStageIndex(state->stage_index) + 1u == SPARK_QWEN38_SERVING_PP_STAGE_COUNT && status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[index];
	}
	pending->active = 0u;
	state->completion_function(state->completion_context,&completion);
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
		status = SparkQwen38ServingUploadBlockTable(state);
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		status = SparkQwen38ServingRunFrame(state,submission,pending,0u,0u,0u,submission->row_count);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
	{
		uint32_t lane,wave,chunk_rows;
		for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
		{
			uint32_t lane_rows;
			lane_rows = 0u;
			for (wave=0u; wave<submission->row_count; wave++)
				lane_rows += submission->row_lane_indices[wave] == lane ? 1u : 0u;
			for (wave=0u; status == SPARK_STATUS_OK && wave<lane_rows; wave+=chunk_rows)
			{
				chunk_rows = lane_rows - wave;
				if ( chunk_rows > state->max_active_sequence_count )
					chunk_rows = state->max_active_sequence_count;
				status = SparkQwen38ServingRunFrame(state,submission,pending,1u,lane,wave,chunk_rows);
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
		pending->active = 0u;
		return(status);
	}
	SparkQwen38ServingCommitSubmission(state,submission);
	SparkQwen38ServingComplete(state,pending,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkQwen38ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkQwen38ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkQwen38ServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkQwen38ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SparkQwen38ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkQwen38ServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkQwen38ServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkQwen38ServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkQwen38ServingAvailableSubmissionCount(state);
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

static void SparkQwen38ServingDestroy(void *adapter_state)
{
	SparkQwen38ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkQwen38ServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkQwen38ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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
	if ( state->device_block_indices != 0 )
		(void)cudaFree(state->device_block_indices);
	if ( state->device_block_counts != 0 )
		(void)cudaFree(state->device_block_counts);
	if ( state->gather_scratch != 0 )
		(void)cudaFree(state->gather_scratch);
	free(state->host_block_indices);
	free(state->free_blocks);
	free(state);
}

static SparkStatus SparkQwen38ServingLoadDriver(
	SparkQwen38ServingState *state,
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
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_QWEN38_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,QWEN38_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_QWEN38_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_QWEN38_SERVING_TARGET) != 0 || strcmp(descriptor->model_description_sha256,QWEN38_CONTRACT_SHA256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || (state->program->flags & SPARK_QWEN38_SERVING_REQUIRED_PROGRAM_FLAGS) != SPARK_QWEN38_SERVING_REQUIRED_PROGRAM_FLAGS || state->program->max_inflight < state->pipeline_slot_count || state->program->profile == 0 || state->program->profile->max_active_slots < state->max_active_sequence_count || state->program->profile->max_new_tokens < state->max_input_row_count )
		return(SPARK_STATUS_TARGET_MISMATCH);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = configuration->node_id;
	request.node_target = configuration->node_target;
	request.kv_logical_page_capacity =
		configuration->runtime_limits.kv_logical_page_capacity;
	request.kv_physical_page_capacity =
		configuration->runtime_limits.kv_physical_page_capacity;
	request.kv_backing_directory = configuration->kv_backing_directory;
	request.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	request.execution_stream = configuration->execution_stream;
	request.completion_function = SparkQwen38ServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkQwen38ServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

static SparkStatus SparkQwen38ServingAllocatePools(
	SparkQwen38ServingState *state)
{
	uint32_t block;
	uint64_t indices;
	indices = (uint64_t)state->max_active_sequence_count * state->blocks_per_lane;
	state->host_block_indices = (uint32_t *)malloc((size_t)indices * sizeof(uint32_t));
	state->free_blocks = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	if ( state->host_block_indices == 0 || state->free_blocks == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (block=0u; block<state->kv_block_count; block++)
		state->free_blocks[block] = state->kv_block_count - 1u - block;
	state->free_block_count = state->kv_block_count;
	if ( cudaMalloc(&state->gather_scratch,(size_t)((uint64_t)state->max_active_sequence_count * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->stage_attn_layer_count != 0u )
	{
		if ( cudaMalloc((void **)&state->device_block_indices,(size_t)(indices * sizeof(uint32_t))) != cudaSuccess || cudaMalloc((void **)&state->device_block_counts,(size_t)(state->max_active_sequence_count * sizeof(uint32_t))) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->block_table.abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->block_table.lane_count = state->max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices;
	state->block_table.lane_physical_block_counts = state->device_block_counts;
	state->block_table.host_physical_block_indices = state->host_block_indices;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkQwen38ServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_QWEN38_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_QWEN38_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
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
	state->first_layer_index = SparkQwen38ServingFirstLayer(configuration->stage_index);
	state->stage_layer_count = SparkQwen38ServingDescriptor.stage_layer_counts[configuration->stage_index];
	state->stage_attn_layer_count = SparkQwen38ServingStageAttentionLayers(state->first_layer_index,state->stage_layer_count);
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
	status = SparkQwen38ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_QWEN38_SERVING_MAX_SEQUENCE_POSITIONS_CAP) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->max_sequence_positions = max_sequence_positions;
		state->blocks_per_lane = (max_sequence_positions + SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		state->kv_block_count = state->resident_sequence_capacity * state->blocks_per_lane;
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
	.progress = SparkQwen38ServingProgress,
	.quiesce = SparkQwen38ServingQuiesce,
	.snapshot = SparkQwen38ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkQwen38ServingInterface);
}
