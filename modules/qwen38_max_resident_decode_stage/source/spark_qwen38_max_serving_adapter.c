/*
 * Qwen 3.6 27B serving adapter: the SparkModelServingAdapterInterface face of
 * the qwen38_resident_decode_stage firmware driver.
 *
 * Two structural differences from the glm52/dsv4 adapters, both owned by the
 * module contract in spark_qwen38_max_resident_decode_stage_firmware.h:
 *
 * - The module is configured through the strict process environment (the
 *   firmware description's runtime_contract lists every variable), not
 *   through a node context struct. The adapter derives the whole slice
 *   environment from its own configuration - stage pack path, PP13 stage
 *   geometry, runtime limits, and the KV pool size implied by the
 *   max_sequence_positions cap - and sets it before driver create. One
 *   resident process hosts one stage, so the process-wide setenv is the
 *   intended channel. SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION is set to 1:
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
 *
 * The adapter body is the shared qwen38 PP-pair body
 * (spark_qwen38_pp_serving_adapter_common.h); this file keeps the family's
 * identity, geometry, capability extras, struct layouts, and the embedded
 * MTP provider slot.
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
#include "sparkpipe/spark_qwen38_max_model.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen38_max_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_speculation_provider.h"

#ifndef QWEN38_MODEL_REVISION
#error "QWEN38_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef QWEN38_CONTRACT_SHA256
#error "QWEN38_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_QWEN38_MAX_SERVING_ADAPTER_ID \
	"spark.qwen38.serving-adapter.tp4-pp4.v1"
#define SPARK_QWEN38_MAX_SERVING_MODEL_ID "Qwen/Qwen3.8-2.4T-A95B"
#define SPARK_QWEN38_MAX_SERVING_DRIVER_MODEL_ID \
	"qwen38.2.4t-a95b.resident-decode-stage-firmware"
#define SPARK_QWEN38_MAX_SERVING_STAGE_NAME "qwen38_resident_decode_stage"
#define SPARK_QWEN38_MAX_SERVING_TARGET \
	"cuda.sm121.qwen38.resident_decode_stage.fp8"
#define SPARK_QWEN38_MAX_SERVING_PROGRAM_NAME "resident_decode"
/* 16 world ranks; stage_index is the world rank. The TP degree is a
 * RUNTIME value from the stage configuration ("tp_degree": 4 -> TP4xPP4,
 * 16 -> TP16xPP1), so one adapter serves both topologies. Pipeline
 * boundaries (hidden in/out, token in/out) are PP boundaries and the
 * per-PP-stage layer counts derive from the degree. The TP-rank tensor
 * sharding (column/row-parallel hidden slices and the router/expert
 * all-reduces) is OUTSTANDING and lands with the rank-local packs. */
#define SPARK_QWEN38_MAX_SERVING_STAGE_COUNT 16u
#define SPARK_QWEN38_MAX_SERVING_DEFAULT_TP_DEGREE 4u
#define SPARK_QWEN38_MAX_SERVING_MAX_PP_STAGE_COUNT 4u
/* Serving caps context at the model's native 262144 until the KV-tier
 * plan lands; the module's KV pool is sized from the deployment's
 * kv_block_count, and the cap merely refuses configs past the model. */
#define SPARK_QWEN38_MAX_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_QWEN38_MAX_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_QWEN38_MAX_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_QWEN38_SERVING_ADAPTER_FN(name) SparkQwen38Max##name
#define SPARK_QWEN38_SERVING_ADAPTER_TYPE(name) SparkQwen38Max##name
#define SPARK_QWEN38_SERVING_ADAPTER_CONST(name) SPARK_QWEN38_MAX_##name
#define SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION QWEN38_MODEL_REVISION
#define SPARK_QWEN38_SERVING_ADAPTER_CONTRACT_SHA256 QWEN38_CONTRACT_SHA256
#define SPARK_QWEN38_SERVING_ADAPTER_TP_DEGREE_VALID(tp_degree) \
	((tp_degree) != 0u && \
	 SPARK_QWEN38_MAX_SERVING_STAGE_COUNT % (tp_degree) == 0u && \
	 (tp_degree) <= SPARK_QWEN38_MAX_SERVING_STAGE_COUNT)
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_COUNT(state) \
	SPARK_QWEN38_MAX_SERVING_STAGE_COUNT
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_INDEX(state) (state)->stage_index
#define SPARK_QWEN38_SERVING_ADAPTER_BIND_FAMILY(state) \
	SparkQwen38MaxServingBindMtpProvider(state)

typedef struct SparkQwen38MaxServingPending
{
	struct SparkQwen38MaxServingState *owner;
	/* The shared submission view (the serving-adapter template fills it). */
	SparkServingAdapterPendingCommon common;
	/* The frame currently inside the driver; completion matches against it. */
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkQwen38MaxServingPending;

/* Per-frame transport shim state. The module calls post_receive/send through
 * the frame context; the shim moves the submission boundary into the frame's
 * expected contiguity. Decode rows are contiguous. Prefill frames are one
 * lane each while the submission boundary is round-major across lanes, so a
 * lane's rows sit at irregular flat offsets whenever lane lengths differ;
 * the row maps give each frame row's flat index in the submission buffer
 * (NULL means the frame rows are contiguous from the base). */
typedef struct SparkQwen38MaxServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkQwen38MaxServingTransportShim;

typedef struct SparkQwen38MaxServingState
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
	uint32_t tp_degree;
	uint32_t pp_stage_count;
	uint32_t stage_layer_counts[SPARK_QWEN38_MAX_SERVING_MAX_PP_STAGE_COUNT];
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
	SparkQwen38MaxKvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkQwen38MaxServingTransportShim shim;
	SparkQwen38MaxServingPending pending[SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	/* The speculation-provider slot (the family's first real use): the
	 * in-checkpoint MTP head behind the embedded-provider shape. The
	 * module's MTP forward FAILS CLOSED today (the module header says so
	 * plainly); the provider carries the contract and renders that fact
	 * with a reason instead of a silent zero. */
	SparkSpeculationProvider provider;
	uint32_t provider_bound;
} SparkQwen38MaxServingState;

/*
 * The embedded MTP provider (the block-drafter binding shape from
 * tests/test_speculation_provider_slot.c, one-token chain). LIFECYCLE +
 * CONTRACT only: the MTP forward stays module-owned and fail-closed, so
 * draft ops refuse with the reason - the design's supports() -> WHY, not
 * a bare UNSUPPORTED.
 */

static SparkStatus SparkQwen38MaxMtpCapabilityQuery(
	const SparkSpeculationGeometryQuery *geometry,
	char *refusal_buffer, uint32_t refusal_buffer_bytes)
{
	if ( geometry == 0 ||
		geometry->hidden_dimension != SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION ||
		geometry->layer_count < SPARK_QWEN38_MAX_MODEL_LAYER_COUNT )
	{
		if ( refusal_buffer != 0 && refusal_buffer_bytes != 0u )
			(void)snprintf(refusal_buffer, refusal_buffer_bytes,
				"qwen38 max mtp provider requires the max target geometry "
				"(hidden %u, %u layers), got hidden %u layers %u",
				SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
				SPARK_QWEN38_MAX_MODEL_LAYER_COUNT,
				geometry != 0 ? geometry->hidden_dimension : 0u,
				geometry != 0 ? geometry->layer_count : 0u);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38MaxMtpDraftBegin(void *provider_state,
	const SparkSpeculationDraftRequest *request)
{
	(void)provider_state;
	(void)request;
	/* fail closed with the reason: the module executes decode only and
	 * its MTP draft forward is UNSUPPORTED (see the module header);
	 * landing it is the provider's first kernel work. */
	return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus SparkQwen38MaxMtpDraftNext(void *provider_state,
	SparkSpeculationDraft *draft)
{
	(void)provider_state;
	(void)draft;
	return(SPARK_STATUS_UNSUPPORTED);
}

static void SparkQwen38MaxMtpDraftCancel(void *provider_state)
{
	(void)provider_state;
}

/* THE one accepted-prefix accounting: MTP drafts one token after the
 * anchor, so a verified chain of N rows commits N-1. */
static SparkStatus SparkQwen38MaxMtpVerifyAccount(void *provider_state,
	uint32_t verified_count, SparkSpeculationVerifyContract *contract_out)
{
	(void)provider_state;
	if ( contract_out == 0 || verified_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(contract_out, 0, sizeof(*contract_out));
	contract_out->chain_width = verified_count;
	contract_out->accepted_token_count = verified_count - 1u;
	contract_out->tokens_per_sequence = contract_out->accepted_token_count;
	contract_out->chain_live = 1u;
	return(SPARK_STATUS_OK);
}

static const SparkSpeculationKvContract SparkQwen38MaxMtpKvContract =
{
	.frame_flags = SPARK_SPECULATION_KV_FLAG_TAIL_FRAME,
	.block_history_depth = 0u
};

static const SparkSpeculationKvContract *SparkQwen38MaxMtpKvContractQuery(
	void *provider_state)
{
	(void)provider_state;
	return(&SparkQwen38MaxMtpKvContract);
}

static const SparkSpeculationProviderOps SparkQwen38MaxMtpProviderOps =
{
	.capability_query = SparkQwen38MaxMtpCapabilityQuery,
	.draft_begin = SparkQwen38MaxMtpDraftBegin,
	.draft_next = SparkQwen38MaxMtpDraftNext,
	.draft_cancel = SparkQwen38MaxMtpDraftCancel,
	.verify_account = SparkQwen38MaxMtpVerifyAccount,
	.kv_contract = SparkQwen38MaxMtpKvContractQuery
};

static const char *const SparkQwen38MaxMtpEnvironmentSchema[] =
{
	"SPEC_METHOD",
	"DRAFT_COUNT"
};

static const SparkSpeculationProviderDescriptor SparkQwen38MaxMtpProviderDescriptor =
{
	.abi_version = SPARK_SPECULATION_PROVIDER_ABI_VERSION,
	.descriptor_bytes = SPARK_SPECULATION_PROVIDER_DESCRIPTOR_BYTES,
	.kind = SPARK_SPECULATION_PROVIDER_MTP,
	.provider_id = "qwen38max.mtp-head.v1",
	.max_draft_token_count = SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT,
	.default_draft_token_count = SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT,
	.environment_schema = SparkQwen38MaxMtpEnvironmentSchema,
	.environment_schema_count = 2u
};

static SparkStatus SparkQwen38MaxServingBindMtpProvider(
	SparkQwen38MaxServingState *state)
{
	SparkStatus status;
	state->provider.descriptor = &SparkQwen38MaxMtpProviderDescriptor;
	state->provider.ops = &SparkQwen38MaxMtpProviderOps;
	state->provider.provider_state = 0;
	status = SparkSpeculationProviderValidate(&state->provider);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->provider_bound = 1u;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterDescriptor SparkQwen38MaxServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_QWEN38_MAX_SERVING_ADAPTER_ID,
		SPARK_QWEN38_MAX_SERVING_MODEL_ID,
		QWEN38_MODEL_REVISION,
		SPARK_QWEN38_MAX_SERVING_PROGRAM_NAME,
		QWEN38_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT),
	.stage_count = SPARK_QWEN38_MAX_SERVING_STAGE_COUNT,
	.layer_count = SPARK_QWEN38_MAX_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_QWEN38_MAX_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	/* The lean module executes every frame on ONE slot (slots[0]) with a
	 * per-frame stream sync, so concurrent inflight frames would share one
	 * set of buffers; advertise 1 until multi-slot pipelining lands. */
	.max_inflight_submission_count = 1u,
	.max_active_sequence_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	/* the in-checkpoint MTP head's envelope behind the provider slot
	 * (one draft layer); the shared validator pins this count to the
	 * SPECULATION capability flag above */
	.max_speculative_token_count = SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.stage_layer_counts = {0u,0u,0u,0u},
	.minimum_efficient_submission_row_count = 0u
};

#include "sparkpipe/spark_qwen38_pp_serving_adapter_common.h"
