/*
 * Qwen 3.6 27B serving adapter: the SparkModelServingAdapterInterface face of
 * the qwen36_resident_decode_stage firmware driver.
 *
 * Two structural differences from the glm52/dsv4 adapters, both owned by the
 * module contract in spark_qwen36_resident_decode_stage_firmware.h:
 *
 * - The module is configured through the strict process environment (the
 *   firmware description's runtime_contract lists every variable), not
 *   through a node context struct. The adapter derives the whole slice
 *   environment from its own configuration - stage pack path, PP13 stage
 *   geometry, runtime limits, and the KV pool size implied by the
 *   max_sequence_positions cap - and sets it before driver create. One
 *   resident process hosts one stage, so the process-wide setenv is the
 *   intended channel. SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION is set to 1:
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
#include "sparkpipe/spark_qwen36_model.h"
#include "spark_qwen36_dspark_format.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen36_serving_adapter.h"

#ifndef QWEN36_MODEL_REVISION
#error "QWEN36_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef QWEN36_CONTRACT_SHA256
#error "QWEN36_CONTRACT_SHA256 must identify the exact package contract"
#endif

/* Serving topology build knob. SPARK_QWEN36_SERVING_TP_DEGREE is the single
 * switch and may be overridden on the compile line (-D...=N):
 *   4 (default) = shipped TP4 whole-stack build (4 TP ranks; unchanged).
 *   1           = TP1 single-rank full-width build.
 *   0           = legacy PP layer-slice build (not shipped).
 * Every downstream constant derives from it; the TP4 default is byte-for-byte
 * the prior build. */
#ifndef SPARK_QWEN36_SERVING_TP_DEGREE
#define SPARK_QWEN36_SERVING_TP_DEGREE 4u
#endif
/* TP mode = single-stage whole stack: every rank runs module stage 1/1 and
 * owns both the embedding and the head; no hidden boundaries. */
#define SPARK_QWEN36_SERVING_TP (SPARK_QWEN36_SERVING_TP_DEGREE >= 1u)
#if SPARK_QWEN36_SERVING_TP_DEGREE == 1u
#define SPARK_QWEN36_SERVING_ADAPTER_ID "spark.qwen36.serving-adapter.tp1.v1"
#define SPARK_QWEN36_SERVING_STAGE_COUNT 1u
#define SPARK_QWEN36_SERVING_STAGE_LAYER_COUNTS {64u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u}
#else
#define SPARK_QWEN36_SERVING_ADAPTER_ID "spark.qwen36.serving-adapter.tp4.v1"
#define SPARK_QWEN36_SERVING_STAGE_COUNT 4u
#define SPARK_QWEN36_SERVING_STAGE_LAYER_COUNTS {64u,64u,64u,64u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u}
#endif
#define SPARK_QWEN36_SERVING_MODEL_ID "Qwen/Qwen3.8-27B"
#define SPARK_QWEN36_SERVING_DRIVER_MODEL_ID \
	"alibaba.qwen3.6-27b.resident-decode-stage-firmware"
#define SPARK_QWEN36_SERVING_STAGE_NAME "qwen36_resident_decode_stage"
#define SPARK_QWEN36_SERVING_TARGET \
	"cuda.sm121.qwen36.resident_decode_stage.bf16"
#define SPARK_QWEN36_SERVING_PROGRAM_NAME "resident_decode"
/* The owner's KV-limit decision: serving caps context at 8192 positions
 * until the long-context KV plan lands, far under the module's 256K admit
 * ceiling. The KV pool is sized from this cap, so a conforming deployment
 * can never exhaust blocks. */
#define SPARK_QWEN36_SERVING_MAX_SEQUENCE_POSITIONS_CAP 8192u
#define SPARK_QWEN36_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

/* MTP chain speculation: when SPARK_QWEN36_SERVING_SPECULATE is set (and not
 * "0"), a decode submission runs the three-frame chain the firmware contract
 * describes - a per-lane decode frame that drafts (MTP_DRAFT_AFTER), a
 * per-lane verify prefill (SPECULATIVE_VERIFY), and a per-lane GDN-restore
 * replay prefill (GDN_RESTORE_FIRST). Disabled, the adapter is the previous
 * non-speculating path unchanged. */
#define SPARK_QWEN36_SERVING_SPECULATE_ENV "SPARK_QWEN36_SERVING_SPECULATE"
#define SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_ENV "SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_COUNT"
/* Draft tokens requested per MTP_DRAFT_AFTER frame. The module caps this at
 * SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS and sizes its draft
 * ids array by the same constant. The verify prefill costs one full-model row
 * walk per draft, so the profitable depth is a tunable: env-overridable via
 * SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_COUNT (1..8). */
#define SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_COUNT \
	SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS

static uint32_t SparkQwen36ServingSpeculativeDraftCount(void)
{
	const char *value = getenv(SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_ENV);
	/* Measured optimum on TP4: D=2 (13.1 tok/s at B1) beats D=1/D=4/D=8 and
	 * the non-spec baseline (12.1). */
	uint32_t count = 2u;
	if ( value != 0 )
	{
		uint32_t parsed = (uint32_t)strtoul(value,0,0);
		if ( parsed >= 1u && parsed <= SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS )
			count = parsed;
	}
	return(count);
}
/* First-draft policy for the MTP chain. draft[0] predicts the just-committed
 * position, so it is redundant with the committed token C0 and is never fed to
 * the verify/replay frames (C0 is fed in its place). "recover" (default)
 * records a first-draft miss as telemetry and keeps speculating; "strict"
 * preserves the legacy behavior (a miss declares the chain dead and zeroes
 * speculation) for A/B comparison. */
#define SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_ENV "SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY"
#define SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_RECOVER 0u
#define SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT 1u
static uint32_t SparkQwen36ServingSpecFirstDraftPolicy(void)
{
	const char *value = getenv(SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_ENV);
	if ( value != 0 && strcmp(value,"strict") == 0 )
		return(SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT);
	return(SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_RECOVER);
}

/* Speculation method: "mtp" (default) drives the MTP chain; "dflash2" drives
 * the DFlash2 block-diffusion drafter (block 8 = C0 anchor + 7 mask tokens,
 * conv-wrapped 5-layer backbone, top-16 + candidate-selector walk). "dspark"
 * names the same driver path for the superseded DSpark drafter pack and fails
 * loudly at load time against a DFlash2 pack (the module geometry check
 * rejects 40-head/FFN-10240 weights). Both block drafters produce draft[0] as
 * the just-committed position (redundant with C0), so the verify/replay
 * phases are shared; only the phase-one draft view, flag, and draft buffer
 * differ. */
#define SPARK_QWEN36_SERVING_SPEC_METHOD_ENV "SPARK_QWEN36_SERVING_SPEC_METHOD"
#define SPARK_QWEN36_SERVING_SPEC_METHOD_MTP 0u
#define SPARK_QWEN36_SERVING_SPEC_METHOD_DSPARK 1u
#define SPARK_QWEN36_SERVING_SPEC_METHOD_DFLASH2 2u
static uint32_t SparkQwen36ServingSpecMethod(void)
{
	const char *value = getenv(SPARK_QWEN36_SERVING_SPEC_METHOD_ENV);
	if ( value != 0 && strcmp(value,"dspark") == 0 )
		return(SPARK_QWEN36_SERVING_SPEC_METHOD_DSPARK);
	if ( value != 0 && strcmp(value,"dflash2") == 0 )
		return(SPARK_QWEN36_SERVING_SPEC_METHOD_DFLASH2);
	return(SPARK_QWEN36_SERVING_SPEC_METHOD_MTP);
}

/* Draft depth for the active spec method: the block drafters always draft
 * their full block_size (verify window = C0 + block-1 drafts); MTP uses the
 * env-tunable depth. */
static uint32_t SparkQwen36ServingBlockDraftMethod(uint32_t spec_method)
{
	return(spec_method == SPARK_QWEN36_SERVING_SPEC_METHOD_DSPARK || spec_method == SPARK_QWEN36_SERVING_SPEC_METHOD_DFLASH2);
}
static uint32_t SparkQwen36ServingActiveDraftCount(uint32_t spec_method)
{
	return(SparkQwen36ServingBlockDraftMethod(spec_method) ? SPARK_QWEN36_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE : SparkQwen36ServingSpeculativeDraftCount());
}
/* GDN snapshot slots. The two-phase min-accept schedule keeps one verify
 * snapshot in flight per lane, capped by the module's slot ceiling; a lane
 * index is the snapshot slot it uses. */
#define SPARK_QWEN36_SERVING_GDN_SNAPSHOT_SLOTS \
	SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS
/* Committed tokens per lane: decode token + up to (D-1) accepted drafts +
 * correction + the replay frame's final emission. */
#define SPARK_QWEN36_SERVING_MAX_COMMITTED_TOKENS \
	(SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS + 2u)

static const char *const SparkQwen36ServingConfigurationMembers[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions"
};

/* Per-lane MTP speculation state, persisted across one submission's three
 * frames (decode-draft, verify, replay). */
typedef struct SparkQwen36ServingSpecState
{
	uint32_t resident_slot;
	uint64_t base_position;
	uint64_t sequence_id;
	uint32_t snapshot_index;
	uint32_t draft_token_count;
	uint32_t accepted_count;
	uint32_t chain_dead;
	uint32_t first_draft_miss;
	uint32_t draft_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	uint32_t emitted_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	uint32_t committed_ids[SPARK_QWEN36_SERVING_MAX_COMMITTED_TOKENS];
} SparkQwen36ServingSpecState;

/* Completion model_extension payload: speculation-reason telemetry so the
 * resident receipt records WHY a first-draft miss happened (and which policy
 * was in effect), not only the accepted-token count. */
#define SPARK_QWEN36_SERVING_EXTENSION_KIND 0x5136u /* "Q6" */
typedef struct SparkQwen36ServingSpecTelemetry
{
	uint32_t first_draft_miss_count;   /* lanes where draft[0] != committed C0 */
	uint32_t first_draft_policy;       /* RECOVER or STRICT (see above) */
} SparkQwen36ServingSpecTelemetry;

typedef struct SparkQwen36ServingPending
{
	struct SparkQwen36ServingState *owner;
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
	uint32_t last_row_by_lane[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t dspark_draft_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	SparkQwen36ServingSpecState spec[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t spec_active;
	uint32_t spec_tokens_per_sequence;
	uint32_t spec_total_accepted;
	uint32_t spec_chain_dead;
	uint32_t spec_first_draft_miss;
} SparkQwen36ServingPending;

/* Per-frame transport shim state. The module calls post_receive/send through
 * the frame context; the shim moves the submission boundary into the frame's
 * expected contiguity. Decode rows are contiguous. Prefill frames are one
 * lane each while the submission boundary is round-major across lanes, so a
 * lane's rows sit at irregular flat offsets whenever lane lengths differ;
 * the row maps give each frame row's flat index in the submission buffer
 * (NULL means the frame rows are contiguous from the base). */
typedef struct SparkQwen36ServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkQwen36ServingTransportShim;

typedef struct SparkQwen36ServingState
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
	uint64_t orphan_completion_count;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkQwen36KvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkQwen36ServingTransportShim shim;
	/* DFlash2 draft source: 0 until the first draft runs; the verify frame
	 * thereafter re-drafts at its tail (state-consistent taps), so the
	 * decode frame stops drafting after the first iteration. Keyed by the
	 * active sequence: a new sequence restarts at the decode frame. */
	uint32_t dflash2_drafts_valid;
	uint64_t dflash2_draft_sequence_id;
	/* Drafts must outlive the submission: the replay-tail drafter writes
	 * here, and the NEXT submission's remap consumes them (the pending
	 * struct dies at the submission boundary). */
	uint32_t dflash2_next_draft_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	SparkQwen36ServingPending pending[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkQwen36ServingState;

static const SparkModelServingAdapterDescriptor SparkQwen36ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION,
	.stage_count = SPARK_QWEN36_SERVING_STAGE_COUNT,
	.layer_count = SPARK_QWEN36_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_QWEN36_SERVING_ADAPTER_ID,
	.model_id = SPARK_QWEN36_SERVING_MODEL_ID,
	.model_revision = QWEN36_MODEL_REVISION,
	.driver_program_name = SPARK_QWEN36_SERVING_PROGRAM_NAME,
	.artifact_sha256 = QWEN36_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_QWEN36_SERVING_STAGE_LAYER_COUNTS,
	.minimum_efficient_submission_row_count = 0u
};

static int32_t SparkQwen36ServingJsonMember(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,root,name));
}

static SparkStatus SparkQwen36ServingJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkQwen36ServingJsonMember(document,root,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkQwen36ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkQwen36ServingState *state,
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
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkQwen36ServingConfigurationMembers,(uint32_t)(sizeof(SparkQwen36ServingConfigurationMembers) / sizeof(SparkQwen36ServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ServingJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_QWEN36_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkQwen36ServingJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,QWEN36_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkQwen36ServingJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ServingJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static uint32_t SparkQwen36ServingFirstLayer(uint32_t stage_index)
{
	uint32_t index,first_layer;
#if SPARK_QWEN36_SERVING_TP
	(void)stage_index;
	return(0u);
#endif
	first_layer = 0u;
	for (index=0u; index<stage_index; index++)
		first_layer += SparkQwen36ServingDescriptor.stage_layer_counts[index];
	return(first_layer);
}

static uint32_t SparkQwen36ServingStageAttentionLayers(uint32_t first_layer, uint32_t layer_count)
{
	uint32_t layer,count;
	count = 0u;
	for (layer=first_layer; layer<first_layer+layer_count; layer++)
		count += SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) == 0u ? 1u : 0u;
	return(count);
}

static uint32_t SparkQwen36ServingOwnsFinalHead(const SparkQwen36ServingState *state);

static uint32_t SparkQwen36ServingSpeculationEnabled(void)
{
	const char *value;
	value = getenv(SPARK_QWEN36_SERVING_SPECULATE_ENV);
	return(value != 0 && value[0] != '\0' && strcmp(value,"0") != 0 ? 1u : 0u);
}

static SparkStatus SparkQwen36ServingSetEnvironment(
	const SparkQwen36ServingState *state)
{
	char value[32];
#define SPARK_QWEN36_SERVING_SET_TEXT(name,text) \
	do { if ( setenv(name,text,1) != 0 ) return(SPARK_STATUS_INTERNAL_ERROR); } while (0)
#define SPARK_QWEN36_SERVING_SET_UNSIGNED(name,number) \
	do { snprintf(value,sizeof(value),"%u",(uint32_t)(number)); SPARK_QWEN36_SERVING_SET_TEXT(name,value); } while (0)
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION","1");
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_PACK_PATH",state->stage_pack_path);
#if SPARK_QWEN36_SERVING_TP
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_COUNT",1u);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_INDEX",0u);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_FIRST_LAYER",0u);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_LAYER_COUNT",SPARK_QWEN36_MODEL_LAYER_COUNT);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_TP_DEGREE",SPARK_QWEN36_SERVING_TP_DEGREE);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_TP_RANK",state->stage_index);
#else
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_COUNT",SPARK_QWEN36_SERVING_STAGE_COUNT);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_INDEX",state->stage_index);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_FIRST_LAYER",state->first_layer_index);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_LAYER_COUNT",state->stage_layer_count);
#endif
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES",state->max_active_sequence_count);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_PIPELINE_SLOTS",state->pipeline_slot_count);
	SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_KV_BLOCKS",state->kv_block_count);
	if ( SparkQwen36ServingSpeculationEnabled() != 0u && SparkQwen36ServingOwnsFinalHead(state) != 0u )
	{
		/* The GDN snapshot is used by BOTH the MTP verify and the block-drafter
		 * verify/replay; only the MTP module itself is suppressed for the
		 * dspark/dflash2 methods. */
		SPARK_QWEN36_SERVING_SET_UNSIGNED("SPARK_QWEN36_STAGE_GDN_SNAPSHOT_SLOTS",SPARK_QWEN36_SERVING_GDN_SNAPSHOT_SLOTS);
		if ( !SparkQwen36ServingBlockDraftMethod(SparkQwen36ServingSpecMethod()) )
			SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_MTP","1");
		else
			SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_MTP","0");
	}
	else
	{
		SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_MTP","0");
		SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_GDN_SNAPSHOT_SLOTS","0");
	}
	/* Fail loudly, never draft silently: a block-drafter method without a
	 * drafter pack initializes an unarmed module whose draft forward is a
	 * no-op (vLLM's V1 trap, mirrored). */
	if ( SparkQwen36ServingSpeculationEnabled() != 0u && SparkQwen36ServingOwnsFinalHead(state) != 0u && SparkQwen36ServingBlockDraftMethod(SparkQwen36ServingSpecMethod()) != 0u )
	{
		const char *drafter_pack = getenv("SPARK_QWEN36_DSPARK_PACK_PATH");
		if ( drafter_pack == 0 || drafter_pack[0] == '\0' )
		{
			fprintf(stderr,"qwen36_serving spec method requires SPARK_QWEN36_DSPARK_PACK_PATH\n");
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	}
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_KV_STORE","none");
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_KV_SERVICE","none");
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_KV_SOCKET","none");
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_KV_POOL_BYTES","0");
	SPARK_QWEN36_SERVING_SET_TEXT("SPARK_QWEN36_STAGE_KV_WORKERS","0");
#undef SPARK_QWEN36_SERVING_SET_TEXT
#undef SPARK_QWEN36_SERVING_SET_UNSIGNED
	return(SPARK_STATUS_OK);
}

/* Wave-major row order, lane bounds, the positions cap, and distinct
 * resident slots. Identical discipline to the glm52 adapter plus the slot
 * uniqueness the qwen36 paged KV table requires: two submission lanes
 * aliasing one resident slot would silently share a KV and GDN state. */
static SparkStatus SparkQwen36ServingRowOrderReject(
	const SparkModelServingSubmission *submission,
	const char *reason)
{
	fprintf(stderr, "qwen36_roworder_reject reason=%s kind=%u rows=%u lanes=%u pos=%llu slot=%u\n",
		reason, submission->work_kind, submission->row_count, submission->active_sequence_count,
		(unsigned long long)(submission->row_count != 0u ? submission->row_positions[0] : 0u),
		submission->active_sequence_count != 0u ? submission->lanes[0].resident_sequence_slot : 0u);
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkQwen36ServingValidateRowOrder(
	const SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint8_t slot_seen[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->resident_sequence_capacity || slot_seen[slot] != 0u )
			return(SparkQwen36ServingRowOrderReject(submission,"slot_range_or_dup"));
		slot_seen[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
			return(SparkQwen36ServingRowOrderReject(submission,"lane_or_position_range"));
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SparkQwen36ServingRowOrderReject(submission,"row_position_gap"));
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SparkQwen36ServingRowOrderReject(submission,"decode_row_count"));
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SparkQwen36ServingRowOrderReject(submission,"wave_order"));
	return(row == submission->row_count ? SPARK_STATUS_OK : SparkQwen36ServingRowOrderReject(submission,"row_count_mismatch"));
}

/* TP stage-position helpers (degree >= 1): every rank owns the embedding and
 * the head and no rank sends or receives hidden boundaries. The legacy PP
 * build (degree 0) keeps the original stage-slice derivations. */
static uint32_t SparkQwen36ServingOwnsEmbedding(const SparkQwen36ServingState *state)
{
	(void)state;
#if SPARK_QWEN36_SERVING_TP
	return(1u);
#else
	return(state->stage_index == 0u ? 1u : 0u);
#endif
}

static uint32_t SparkQwen36ServingOwnsFinalHead(const SparkQwen36ServingState *state)
{
	(void)state;
#if SPARK_QWEN36_SERVING_TP
	return(1u);
#else
	return(state->stage_index + 1u == SPARK_QWEN36_SERVING_STAGE_COUNT ? 1u : 0u);
#endif
}

static uint32_t SparkQwen36ServingNeedsHiddenOutput(const SparkQwen36ServingState *state)
{
	(void)state;
#if SPARK_QWEN36_SERVING_TP
	return(0u);
#else
	return(state->stage_index + 1u < SPARK_QWEN36_SERVING_STAGE_COUNT ? 1u : 0u);
#endif
}

/* Hidden boundary pointers exist only after the resident commits a route:
 * the wire submission validate_submission sees always has them absent (the
 * serving header documents this), so this check is meaningful only from
 * submit, never from validate_submission. */
static SparkStatus SparkQwen36ServingValidateBoundaries(
	const SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES;
	if ( (SparkQwen36ServingOwnsEmbedding(state) == 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (SparkQwen36ServingOwnsEmbedding(state) != 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (SparkQwen36ServingNeedsHiddenOutput(state) != 0u && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (SparkQwen36ServingNeedsHiddenOutput(state) == 0u && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingValidateSubmissionBase(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkQwen36ServingDescriptor,&state->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"qwen36_debug validate_runtime status=%d kind=%u rows=%u lanes=%u act=%u tps=%u new_tokens=%u pos=%llu ctx=%llu\\n",(int)status,submission->work_kind,submission->row_count,submission->lane_count,submission->active_sequence_count,submission->tokens_per_sequence,submission->new_token_count,(unsigned long long)submission->sequence_position,(unsigned long long)(submission->active_sequence_count > 0u ? submission->lanes[0].context_token_count : 0u));
		return(status);
	}
	if ( submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkQwen36ServingValidateRowOrder(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"qwen36_debug row_order status=%d\\n",(int)status);
		return(status);
	}
	if ( submission->model_extension_bytes != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen36ServingState *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SparkQwen36ServingState *)adapter_state;
	status = SparkQwen36ServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}

static SparkQwen36ServingPending *SparkQwen36ServingReservePending(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen36ServingPending *pending;
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

static void SparkQwen36ServingOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkQwen36ServingState *state;
	(void)driver_completion;
	state = (SparkQwen36ServingState *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SparkQwen36ServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkQwen36ServingPending *pending;
	SparkQwen36ServingState *state;
	uint32_t matches;
	pending = (SparkQwen36ServingPending *)completion_context;
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

static void SparkQwen36ServingDriverWake(void *wake_context)
{
	SparkQwen36ServingState *state;
	state = (SparkQwen36ServingState *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SparkQwen36ServingAvailableSubmissionCount(
	const SparkQwen36ServingState *state)
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
static void SparkQwen36ServingReleaseLane(
	SparkQwen36ServingState *state,
	uint32_t slot)
{
	uint32_t ordinal;
	for (ordinal=0u; ordinal<state->lane_block_counts[slot]; ordinal++)
		state->free_blocks[state->free_block_count++] = state->host_block_indices[((uint64_t)slot * state->blocks_per_lane) + ordinal];
	state->lane_block_counts[slot] = 0u;
	state->lane_context_tokens[slot] = 0u;
}

static SparkStatus SparkQwen36ServingCoverLane(
	SparkQwen36ServingState *state,
	uint32_t slot,
	uint64_t end_position)
{
	uint32_t required,ordinal;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	required = (uint32_t)((end_position + SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
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

static void SparkQwen36ServingDropSubmission(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		SparkQwen36ServingReleaseLane(state,submission->lanes[lane].resident_sequence_slot);
}

static SparkStatus SparkQwen36ServingCoverSubmission(
	SparkQwen36ServingState *state,
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
			SparkQwen36ServingReleaseLane(state,slot);
		status = SparkQwen36ServingCoverLane(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
		{
			/* Coverage failure is KV exhaustion: drop every lane the
			 * submission touches so a partial allocation cannot linger. */
			SparkQwen36ServingDropSubmission(state,submission);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen36ServingCommitSubmission(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	const SparkQwen36ServingPending *pending)
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

static SparkStatus SparkQwen36ServingUploadBlockTable(
	const SparkQwen36ServingState *state)
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

/* Extend KV coverage for the speculation chain: the MTP draft rows land at
 * [base_position, base_position + D - 1) and the verify/replay prefills walk
 * through base_position + D + 1, so a speculating lane must hold D + 2 more
 * positions than the plain decode row. Feasibility is proven before any block
 * is handed out, so a refused extension falls back to the plain path cleanly. */
static SparkStatus SparkQwen36ServingExtendSpeculativeCoverage(
	SparkQwen36ServingState *state,
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
		end_position = position + (uint64_t)SparkQwen36ServingActiveDraftCount(state->spec_method) + 2u;
		required = (end_position + SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
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
		end_position = position + (uint64_t)SparkQwen36ServingActiveDraftCount(state->spec_method) + 2u;
		status = SparkQwen36ServingCoverLane(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingPostReceive(
	SparkHiddenTransportSession *transport_session,
	SparkHiddenTransportPacket *packet)
{
	SparkQwen36ServingTransportShim *shim;
	const void *source;
	uint32_t row;
	shim = (SparkQwen36ServingTransportShim *)transport_session;
	if ( shim == 0 || packet == 0 || shim->input_base == 0 || shim->input_rows == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	source = shim->input_base;
	if ( shim->input_row_map != 0 )
	{
		for (row=0u; row<shim->input_rows; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->input_scratch + ((uint64_t)row * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES),(const uint8_t *)shim->input_base + ((uint64_t)shim->input_row_map[row] * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES),SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)shim->execution_stream) != cudaSuccess )
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
	packet->hidden_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES;
	packet->hidden_bf16 = source;
	packet->cuda_stream = shim->execution_stream;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingSend(
	SparkHiddenTransportSession *transport_session,
	const SparkHiddenTransportPacket *packet)
{
	SparkQwen36ServingTransportShim *shim;
	uint32_t row;
	shim = (SparkQwen36ServingTransportShim *)transport_session;
	if ( shim == 0 || packet == 0 || packet->hidden_bf16 == 0 || shim->output_base == 0 || packet->active_sequence_count == 0u || packet->hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( shim->output_row_map != 0 )
	{
		for (row=0u; row<packet->active_sequence_count; row++)
			if ( cudaMemcpyAsync((uint8_t *)shim->output_base + ((uint64_t)shim->output_row_map[row] * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES),(const uint8_t *)packet->hidden_bf16 + ((uint64_t)row * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES),SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
				return(SPARK_STATUS_IO_ERROR);
	}
	else if ( cudaMemcpyAsync(shim->output_base,packet->hidden_bf16,(uint64_t)packet->active_sequence_count * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)packet->cuda_stream) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	return(cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static void SparkQwen36ServingBuildFrame(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen36ServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows,
	SparkQwen36DecodeBatchView *decode_batch,
	SparkQwen36PrefillFrameView *prefill_view,
	SparkQwen36ResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t slot;
	uint64_t base_position;
	uint32_t row;
	slot = prefill != 0u ? pending->resident_slots[lane] : 0u;
	base_position = 0u;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
		context->kv_block_table = &state->block_table;
	}
	if ( SparkQwen36ServingOwnsEmbedding(state) == 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen36ServingPostReceive;
	}
	if ( SparkQwen36ServingNeedsHiddenOutput(state) != 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SparkQwen36ServingSend;
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
		prefill_view->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = submission->row_sequence_ids[pending->frame_row_flats[0]];
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW;
		context->prefill_frame = prefill_view;
	}
	else
	{
		for (row=0u; row<frame_rows; row++)
			pending->frame_row_slots[row] = pending->resident_slots[submission->row_lane_indices[row]];
		memcpy(pending->frame_token_ids,submission->token_ids,(size_t)frame_rows * sizeof(uint32_t));
		state->shim.input_row_map = 0;
		state->shim.output_row_map = 0;
		decode_batch->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = submission->row_positions;
		decode_batch->row_sequence_ids = submission->row_sequence_ids;
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
		context->decode_batch = decode_batch;
	}
	memset(buffers,0,sizeof(SparkModelDriverBuffer[2]));
	if ( SparkQwen36ServingOwnsEmbedding(state) != 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen36ServingOwnsFinalHead(state) != 0u )
	{
		uint32_t out_index;
		out_index = SparkQwen36ServingOwnsEmbedding(state) != 0u ? 1u : 0u;
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
	frame->buffers = SparkQwen36ServingOwnsEmbedding(state) != 0u || SparkQwen36ServingOwnsFinalHead(state) != 0u ? buffers : 0;
	frame->buffer_count = (SparkQwen36ServingOwnsEmbedding(state) != 0u ? 1u : 0u) + (SparkQwen36ServingOwnsFinalHead(state) != 0u ? 1u : 0u);
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkQwen36ServingDriverCompletion;
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SparkQwen36ServingAdmit(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	(void)submission;
	status = SparkAdmissionRequestFromFrame(
		state->program->program_id,frame,0,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluateAndApply(
		state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SparkQwen36ServingRunFrame(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen36ServingPending *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows)
{
	SparkQwen36DecodeBatchView decode_batch;
	SparkQwen36PrefillFrameView prefill_view;
	SparkQwen36ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SparkQwen36ServingBuildFrame(state,submission,pending,prefill,lane,wave_base,frame_rows,&decode_batch,&prefill_view,&context,buffers,&frame);
	status = SparkQwen36ServingAdmit(state,submission,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen36_admit_reject status=%d prefill=%u frame_rows=%u lanes=%u\n", (int)status, prefill, frame_rows, submission->active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen36_debug driver_submit status=%d prefill=%u rows=%u seqpos=%llu tps=%u newtok=%u\n", (int)status, prefill, frame_rows, (unsigned long long)submission->sequence_position, submission->tokens_per_sequence, submission->new_token_count);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen36_debug frame_status status=%d prefill=%u rows=%u\n", (int)status, prefill, frame_rows);
	if ( status == SPARK_STATUS_OK && SparkQwen36ServingOwnsFinalHead(state) != 0u )
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

/* Build one speculative frame: a single-lane decode with MTP_DRAFT_AFTER, or a
 * single-lane prefill with SPECULATIVE_VERIFY / GDN_RESTORE_FIRST. The token
 * rows are caller-owned host ids (the lane's decode token or the drafts), not
 * gathered from the submission, and the output buffer is sized for the exact
 * id count the module contract emits. */
static void SparkQwen36ServingBuildSpeculativeFrame(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen36ServingPending *pending,
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
	SparkQwen36MtpDraftView *mtp_draft,
	SparkQwen36DsparkDraftView *dspark_draft,
	SparkQwen36GdnSnapshotView *gdn_snapshot,
	uint32_t output_id_count,
	SparkQwen36DecodeBatchView *decode_batch,
	SparkQwen36PrefillFrameView *prefill_view,
	SparkQwen36ResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t out_index;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
		context->kv_block_table = &state->block_table;
	}
	if ( SparkQwen36ServingOwnsEmbedding(state) == 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SparkQwen36ServingPostReceive;
	}
	if ( SparkQwen36ServingNeedsHiddenOutput(state) != 0u )
	{
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SparkQwen36ServingSend;
	}
	state->shim.input_base = submission->hidden_input_address;
	state->shim.input_rows = frame_rows;
	state->shim.input_row_map = 0;
	state->shim.output_base = submission->hidden_output_address;
	state->shim.output_row_map = 0;
	memcpy(pending->frame_token_ids,token_ids,(size_t)frame_rows * sizeof(uint32_t));
	if ( prefill != 0u )
	{
		prefill_view->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION;
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = frame_sequence_id;
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW;
		context->prefill_frame = prefill_view;
	}
	else
	{
		pending->frame_row_slots[0] = slot;
		decode_batch->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = row_positions;
		decode_batch->row_sequence_ids = row_sequence_ids;
		context->flags |= SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
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
	if ( SparkQwen36ServingOwnsEmbedding(state) != 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SparkQwen36ServingOwnsFinalHead(state) != 0u )
	{
		out_index = SparkQwen36ServingOwnsEmbedding(state) != 0u ? 1u : 0u;
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
	frame->buffers = SparkQwen36ServingOwnsEmbedding(state) != 0u || SparkQwen36ServingOwnsFinalHead(state) != 0u ? buffers : 0;
	frame->buffer_count = (SparkQwen36ServingOwnsEmbedding(state) != 0u ? 1u : 0u) + (SparkQwen36ServingOwnsFinalHead(state) != 0u ? 1u : 0u);
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkQwen36ServingDriverCompletion;
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SparkQwen36ServingRunSpeculativeFrame(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen36ServingPending *pending,
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
	SparkQwen36MtpDraftView *mtp_draft,
	SparkQwen36DsparkDraftView *dspark_draft,
	SparkQwen36GdnSnapshotView *gdn_snapshot,
	uint32_t output_id_count)
{
	SparkQwen36DecodeBatchView decode_batch;
	SparkQwen36PrefillFrameView prefill_view;
	SparkQwen36ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SparkQwen36ServingBuildSpeculativeFrame(state,submission,pending,slot,prefill,token_ids,row_positions,row_sequence_ids,frame_rows,base_position,frame_sequence_id,frame_sequence_position,extra_flags,mtp_draft,dspark_draft,gdn_snapshot,output_id_count,&decode_batch,&prefill_view,&context,buffers,&frame);
	status = SparkQwen36ServingAdmit(state,submission,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "qwen36_admit_reject status=%d prefill=%u frame_rows=%u lanes=%u\n", (int)status, prefill, frame_rows, submission->active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	return(status);
}

/* Chain speculative decode for one decode submission. Precondition: the head
 * stage owns the head, speculation is armed, active_sequence_count fits the
 * GDN snapshot slots, and the draft-chain KV coverage was extended.
 *
 * Phase one runs each lane's MTP_DRAFT_AFTER decode frame (committed token +
 * D drafts) and its SPECULATIVE_VERIFY prefill (D emitted ids), accepting the
 * leading matches host-side: emitted[i] == draft[i+1]. Phase two runs each
 * lane's GDN_RESTORE_FIRST replay over the accepted drafts plus the correction
 * token, whose final emission is the next committed token. Every lane commits
 * the same min_accepted depth so the completion stays lane-uniform. */
static SparkStatus SparkQwen36ServingSubmitSpeculativeDecode(
	SparkQwen36ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkQwen36ServingPending *pending)
{
	SparkQwen36MtpDraftView mtp_draft;
	SparkQwen36DsparkDraftView dspark_draft;
	SparkQwen36GdnSnapshotView gdn_snapshot;
	uint32_t lane,draft;
	uint32_t draft_count;
	uint32_t min_accepted;
	uint32_t first_draft_policy;
	uint32_t spec_method;
	uint32_t verify_tokens[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
	SparkStatus status;
	spec_method = state->spec_method;
	draft_count = SparkQwen36ServingActiveDraftCount(state->spec_method);
	first_draft_policy = SparkQwen36ServingSpecFirstDraftPolicy();
	pending->spec_active = 1u;
	pending->spec_first_draft_miss = 0u;
	memset(pending->spec,0,sizeof(pending->spec));
	status = SPARK_STATUS_OK;
	for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
	{
		SparkQwen36ServingSpecState *spec;
		uint32_t slot,last_row;
		uint64_t position,sequence;
		uint32_t token;
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
		if ( SparkQwen36ServingBlockDraftMethod(spec_method) )
		{
			{
				/* Draft on EVERY decode frame: the anchor must be this round's
				 * C0 = the decode's own emission (the oracle-verified winner;
				 * at a replay tail output_token_ids holds the replay emission,
				 * which drafts from the wrong token). */
			
				memset(&dspark_draft,0,sizeof(dspark_draft));
				dspark_draft.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
				dspark_draft.descriptor_bytes = sizeof(dspark_draft);
				dspark_draft.block_size = draft_count;
				/* DFlash2 emits block-1 draft ids (the mask slots); DSpark emitted
				 * one per block row and the remap below dropped the last. */
				dspark_draft.draft_token_count = spec_method == SPARK_QWEN36_SERVING_SPEC_METHOD_DFLASH2 ? draft_count - 1u : draft_count;
				dspark_draft.sequence_id = sequence;
				dspark_draft.base_position = spec->base_position;
				dspark_draft.tap_buffer = 0;
				dspark_draft.draft_token_ids = state->dflash2_next_draft_ids;
				status = SparkQwen36ServingRunSpeculativeFrame(state,submission,pending,slot,0u,&token,&position,&sequence,1u,0u,submission->sequence_id,submission->sequence_position,SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER,0,&dspark_draft,0,1u);
			}
		}
		else
		{
			memset(&mtp_draft,0,sizeof(mtp_draft));
			mtp_draft.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION;
			mtp_draft.descriptor_bytes = sizeof(mtp_draft);
			mtp_draft.lane_index = slot;
			mtp_draft.draft_token_count = draft_count;
			mtp_draft.base_position = spec->base_position;
			mtp_draft.sequence_id = sequence;
			mtp_draft.row_token_ids = pending->frame_token_ids;
			status = SparkQwen36ServingRunSpeculativeFrame(state,submission,pending,slot,0u,&token,&position,&sequence,1u,0u,submission->sequence_id,submission->sequence_position,SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER,&mtp_draft,0,0,1u + draft_count);
		}
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "qwen36_spec_diag decode_frame_failed lane=%u status=%d\n", lane, (int)status);
		if ( status == SPARK_STATUS_OK )
		{
			spec->committed_ids[0] = pending->frame_output_ids[0];
			if ( SparkQwen36ServingBlockDraftMethod(spec_method) )
			{
				/* Block drafters draft position C0+i+1 from block[i+1] (block[0]
				 * = embed(C0)), while the shared verify/replay/accept loop below
				 * uses the MTP convention: draft[0] predicts C0 (redundant, never
				 * verified) and draft[i>=1] predicts position C0+i. Remap so
				 * draft_ids[0]=C0 (first-draft always in agreement) and
				 * draft_ids[i]=block_draft[i-1] for i>=1. DSpark's last draft
				 * (position C0+7) drops out exactly like MTP's redundant
				 * draft[0]; DFlash2's block-1 ids all land. */
				spec->draft_ids[0] = spec->committed_ids[0];
				for (draft=1u; draft<draft_count; draft++)
					spec->draft_ids[draft] = state->dflash2_next_draft_ids[draft - 1u];
			}
			else
			{
				for (draft=0u; draft<draft_count; draft++)
					spec->draft_ids[draft] = pending->frame_output_ids[1u + draft];
			}
			/* draft[0] predicts the just-committed position, so it is redundant
			 * with C0 and is never fed to verify/replay (C0 is fed in its place).
			 * A first-draft miss is recorded in telemetry + the receipt; only the
			 * strict policy turns it into a dead chain (legacy zero-speculation). */
			spec->first_draft_miss = spec->draft_ids[0] != spec->committed_ids[0] ? 1u : 0u;
			spec->chain_dead = (spec->first_draft_miss != 0u && first_draft_policy == SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT) ? 1u : 0u;
			if ( spec->first_draft_miss != 0u )
				fprintf(stderr, "qwen36_spec first_draft_miss lane=%u C0=%u draft0=%u policy=%s\n", lane, spec->committed_ids[0], spec->draft_ids[0], first_draft_policy == SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY_STRICT ? "strict" : "recover");
		}
		if ( status == SPARK_STATUS_OK && spec->chain_dead == 0u )
		{
			memset(&gdn_snapshot,0,sizeof(gdn_snapshot));
			gdn_snapshot.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
			gdn_snapshot.descriptor_bytes = sizeof(gdn_snapshot);
			gdn_snapshot.snapshot_index = spec->snapshot_index;
			/* Feed C0 (not draft[0]) as the first verify row: draft[0]
			 * predicts the already-committed position, so it is redundant
			 * and a first-draft miss must not poison the rest of the chain. */
			verify_tokens[0] = spec->committed_ids[0];
			for (draft=1u; draft<draft_count; draft++)
				verify_tokens[draft] = spec->draft_ids[draft];
			status = SparkQwen36ServingRunSpeculativeFrame(state,submission,pending,slot,1u,verify_tokens,0,0,draft_count,spec->base_position,sequence,spec->base_position,SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY,0,0,&gdn_snapshot,draft_count);
			if ( status != SPARK_STATUS_OK )
				fprintf(stderr, "qwen36_spec_diag verify_frame_failed lane=%u status=%d\n", lane, (int)status);
		}
		if ( status == SPARK_STATUS_OK )
		{
			for (draft=0u; draft<draft_count; draft++)
				spec->emitted_ids[draft] = pending->frame_output_ids[draft];
			spec->accepted_count = 0u;
			while ( spec->accepted_count + 1u < draft_count && spec->emitted_ids[spec->accepted_count] == spec->draft_ids[spec->accepted_count + 1u] )
				spec->accepted_count++;
			fprintf(stderr, "qwen36_spec_diag C0=%u accepted=%u drafts=[%u,%u,%u,%u] emitted=[%u,%u,%u,%u]\n",
				spec->committed_ids[0], spec->accepted_count,
				spec->draft_ids[0], spec->draft_ids[1], spec->draft_ids[2], spec->draft_ids[3],
				spec->emitted_ids[0], spec->emitted_ids[1], spec->emitted_ids[2], spec->emitted_ids[3]);
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
		/* A dead chain's verify output is poisoned, so a batch with any dead
		 * lane commits the model token alone for every lane (speculation is
		 * simply not credited this round; tokens stay exact). */
		if ( pending->spec_chain_dead != 0u )
			min_accepted = 0u;
		/* The shared serving ABI caps tokens_per_sequence at
		 * SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE (8); a fully-accepted
		 * block-8 chain would commit 10 (C0 + 7 drafts + correction +
		 * replay emission). Clamp acceptance so the completion fits the cap;
		 * the two surplus verified drafts are discarded and re-drafted next
		 * iteration. Removing the clamp needs the shared ABI bump, reviewed
		 * cross-session. */
		if ( min_accepted + 3u > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE )
			min_accepted = SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE - 3u;
		pending->spec_tokens_per_sequence = pending->spec_chain_dead != 0u ? 1u : min_accepted + 3u;
		pending->spec_total_accepted = pending->spec_chain_dead != 0u ? 0u : min_accepted * submission->active_sequence_count;
	}
	for (lane=0u; status == SPARK_STATUS_OK && pending->spec_chain_dead == 0u && lane<submission->active_sequence_count; lane++)
	{
		SparkQwen36ServingSpecState *spec;
		uint32_t slot;
		uint32_t replay_rows;
		/* +1: a fully-accepted block-8 chain replays C0 + 7 drafts + the
		 * correction = 9 rows, one past MAX_MTP_DRAFT_TOKENS. */
		uint32_t replay_tokens[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS + 1u];
		uint64_t replay_base;
		spec = &pending->spec[lane];
		slot = spec->resident_slot;
		/* The snapshot restores the GDN state to BEFORE the first drafted
		 * position, so the replay must re-walk it too: the committed token
		 * C0, the accepted drafts, and the correction. draft[0] is not used
		 * (it predicted the already-committed position). */
		replay_rows = min_accepted + 2u;
		replay_tokens[0] = spec->committed_ids[0];
		for (draft=1u; draft<=min_accepted; draft++)
			replay_tokens[draft] = spec->draft_ids[draft];
		replay_tokens[min_accepted + 1u] = spec->emitted_ids[min_accepted];
		replay_base = spec->base_position;
		memset(&gdn_snapshot,0,sizeof(gdn_snapshot));
		gdn_snapshot.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION;
		gdn_snapshot.descriptor_bytes = sizeof(gdn_snapshot);
		gdn_snapshot.snapshot_index = spec->snapshot_index;
		status = SparkQwen36ServingRunSpeculativeFrame(state,submission,pending,slot,1u,replay_tokens,0,0,replay_rows,replay_base,spec->sequence_id,replay_base,SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST,0,0,&gdn_snapshot,1u);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "qwen36_spec_diag replay_frame_failed lane=%u status=%d\n", lane, (int)status);
		if ( status == SPARK_STATUS_OK )
		{
			for (draft=0u; draft<min_accepted; draft++)
				spec->committed_ids[1u + draft] = spec->draft_ids[1u + draft];
			spec->committed_ids[1u + min_accepted] = spec->emitted_ids[min_accepted];
			spec->committed_ids[2u + min_accepted] = pending->frame_output_ids[0];
		}
	}
	return(status);
}

static void SparkQwen36ServingComplete(
	SparkQwen36ServingState *state,
	SparkQwen36ServingPending *pending,
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
	if ( SparkQwen36ServingOwnsFinalHead(state) != 0u && status == SPARK_STATUS_OK )
	{
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		if ( pending->spec_active != 0u )
		{
			uint32_t lane,step;
			completion.tokens_per_sequence = pending->spec_tokens_per_sequence;
			completion.token_count = pending->active_sequence_count * pending->spec_tokens_per_sequence;
			for (lane=0u; lane<pending->active_sequence_count; lane++)
				for (step=0u; step<completion.tokens_per_sequence; step++)
					completion.token_ids[(lane * completion.tokens_per_sequence) + step] = pending->spec[lane].committed_ids[step];
		}
		else
		{
			completion.tokens_per_sequence = 1u;
			completion.token_count = pending->active_sequence_count;
			for (index=0u; index<completion.token_count; index++)
				completion.token_ids[index] = pending->output_token_ids[index];
		}
	}
	if ( pending->spec_active != 0u )
	{
		SparkQwen36ServingSpecTelemetry telemetry;
		memset(&telemetry,0,sizeof(telemetry));
		telemetry.first_draft_miss_count = pending->spec_first_draft_miss;
		telemetry.first_draft_policy = SparkQwen36ServingSpecFirstDraftPolicy();
		completion.completion_flags |= SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION;
		completion.model_extension_kind = SPARK_QWEN36_SERVING_EXTENSION_KIND;
		completion.model_extension_bytes = sizeof(telemetry);
		memcpy(completion.model_extension,&telemetry,sizeof(telemetry));
	}
	pending->active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static SparkStatus SparkQwen36ServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkQwen36ServingState *state;
	SparkQwen36ServingPending *pending;
	SparkStatus status;
	uint32_t speculate;
	state = (SparkQwen36ServingState *)adapter_state;
	status = SparkQwen36ServingValidateSubmissionBase(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		status = SparkQwen36ServingValidateBoundaries(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "qwen36_submit_reject status=%d kind=%u rows=%u lanes=%u pos=%llu slot=%u\n", (int)status, submission->work_kind, submission->row_count, submission->active_sequence_count, (unsigned long long)(submission->row_count != 0u ? submission->row_positions[0] : 0u), submission->row_count != 0u ? submission->lanes[0].resident_sequence_slot : 0u);
		return(status);
	}
	pending = SparkQwen36ServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	speculate = 0u;
	status = SparkQwen36ServingCoverSubmission(state,submission);
	/* B1 only: the per-lane chain is serial by contract, so batched decodes
	 * (B2+) would serialize D+2 extra full-model walks per lane and lose to
	 * the plain batched path (measured). */
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE && SparkQwen36ServingSpeculationEnabled() != 0u && SparkQwen36ServingOwnsFinalHead(state) != 0u && submission->active_sequence_count == 1u )
	{
		status = SparkQwen36ServingExtendSpeculativeCoverage(state,submission);
		if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
		{
			/* The KV pool cannot hold the draft chain (near the context cap):
			 * fall back to the plain batched decode. */
			speculate = 0u;
			status = SPARK_STATUS_OK;
		}
		else if ( status == SPARK_STATUS_OK )
			speculate = 1u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ServingUploadBlockTable(state);
	if ( status == SPARK_STATUS_OK && speculate != 0u )
		status = SparkQwen36ServingSubmitSpeculativeDecode(state,submission,pending);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		status = SparkQwen36ServingRunFrame(state,submission,pending,0u,0u,0u,submission->row_count);
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
				status = SparkQwen36ServingRunFrame(state,submission,pending,1u,lane,wave,chunk_rows);
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
		SparkQwen36ServingDropSubmission(state,submission);
		pending->active = 0u;
		return(status);
	}
	if ( pending->spec_active != 0u )
	{
		pending->accepted_token_count = pending->spec_total_accepted;
		fprintf(stderr,"qwen36_spec accepted=%u\n",pending->spec_total_accepted);
	}
	SparkQwen36ServingCommitSubmission(state,submission,pending);
	SparkQwen36ServingComplete(state,pending,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkQwen36ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkQwen36ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkQwen36ServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkQwen36ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SparkQwen36ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkQwen36ServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkQwen36ServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkQwen36ServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkQwen36ServingAvailableSubmissionCount(state);
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

static void SparkQwen36ServingDestroy(void *adapter_state)
{
	SparkQwen36ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkQwen36ServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkQwen36ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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

static SparkStatus SparkQwen36ServingLoadDriver(
	SparkQwen36ServingState *state,
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
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_QWEN36_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,QWEN36_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_QWEN36_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_QWEN36_SERVING_TARGET) != 0 || strcmp(descriptor->model_description_sha256,QWEN36_CONTRACT_SHA256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || (state->program->flags & SPARK_QWEN36_SERVING_REQUIRED_PROGRAM_FLAGS) != SPARK_QWEN36_SERVING_REQUIRED_PROGRAM_FLAGS || state->program->max_inflight < state->pipeline_slot_count || state->program->profile == 0 || state->program->profile->max_active_slots < state->max_active_sequence_count || state->program->profile->max_new_tokens < state->max_input_row_count )
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
	request.completion_function = SparkQwen36ServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkQwen36ServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

static SparkStatus SparkQwen36ServingAllocatePools(
	SparkQwen36ServingState *state)
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
	if ( cudaMalloc(&state->gather_scratch,(size_t)((uint64_t)state->max_active_sequence_count * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->stage_attn_layer_count != 0u )
	{
		if ( cudaMalloc((void **)&state->device_block_indices,(size_t)(indices * sizeof(uint32_t))) != cudaSuccess || cudaMalloc((void **)&state->device_block_counts,(size_t)(state->max_active_sequence_count * sizeof(uint32_t))) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->block_table.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->block_table.lane_count = state->max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices;
	state->block_table.lane_physical_block_counts = state->device_block_counts;
	state->block_table.host_physical_block_indices = state->host_block_indices;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkQwen36ServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_QWEN36_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_QWEN36_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkQwen36ServingState *state;
	uint32_t max_sequence_positions;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkQwen36ServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkQwen36ServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->stage_index = configuration->stage_index;
	state->first_layer_index = SparkQwen36ServingFirstLayer(configuration->stage_index);
	state->stage_layer_count = SparkQwen36ServingDescriptor.stage_layer_counts[configuration->stage_index];
	state->stage_attn_layer_count = SparkQwen36ServingStageAttentionLayers(state->first_layer_index,state->stage_layer_count);
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
	state->spec_method = SparkQwen36ServingSpecMethod();
	status = SparkQwen36ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_QWEN36_SERVING_MAX_SEQUENCE_POSITIONS_CAP) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->max_sequence_positions = max_sequence_positions;
		state->blocks_per_lane = (max_sequence_positions + SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		state->kv_block_count = state->resident_sequence_capacity * state->blocks_per_lane;
		status = SparkQwen36ServingAllocatePools(state);
		state->shim.input_scratch = state->gather_scratch;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ServingSetEnvironment(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ServingLoadDriver(state,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen36ServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface SparkQwen36ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkQwen36ServingDescriptor,
	.initialize = SparkQwen36ServingInitialize,
	.destroy = SparkQwen36ServingDestroy,
	.validate_submission = SparkQwen36ServingValidateSubmission,
	.submit = SparkQwen36ServingSubmit,
	.progress = SparkQwen36ServingProgress,
	.quiesce = SparkQwen36ServingQuiesce,
	.snapshot = SparkQwen36ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkQwen36ServingInterface);
}
