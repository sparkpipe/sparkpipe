#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_mimo26_model.h"
#include "sparkpipe/spark_mimo26_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_mimo26_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#define SPARK_FAMILY_CAMEL Mimo26
#define SPARK_FAMILY_UPPER MIMO26
#define SPARK_FAMILY_LOWER mimo26

#include "sparkpipe/family/spark_family.h"

#ifndef MIMO26_MODEL_REVISION
#error "MIMO26_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef MIMO26_CONTRACT_SHA256
#error "MIMO26_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_MIMO26_SERVING_ADAPTER_ID "spark.mimo26.serving-adapter.tp4.v1"
#define SPARK_MIMO26_SERVING_MODEL_ID "XiaomiMiMo/MiMo-V2.6-Flash"
#define SPARK_MIMO26_SERVING_DRIVER_MODEL_ID "mimo26.resident-decode-stage-firmware"
#define SPARK_MIMO26_SERVING_STAGE_NAME "mimo26_resident_decode_stage"
#define SPARK_MIMO26_SERVING_TARGET "cuda.sm121.mimo26.resident_decode_stage.mxfp4"
#define SPARK_MIMO26_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_MIMO26_SERVING_STAGE_COUNT 4u
#define SPARK_MIMO26_SERVING_DEFAULT_TP_DEGREE 4u
#define SPARK_MIMO26_SERVING_PARALLEL_GROUP_SIZE 4u
#define SPARK_MIMO26_SERVING_PP_STAGE_COUNT 1u
#define SPARK_MIMO26_SERVING_MAX_PP_STAGE_COUNT SPARK_MIMO26_SERVING_PP_STAGE_COUNT
#define SPARK_MIMO26_SERVING_STAGE_LAYER_LIST SPARK_MIMO26_MODEL_LAYER_COUNT,SPARK_MIMO26_MODEL_LAYER_COUNT,SPARK_MIMO26_MODEL_LAYER_COUNT,SPARK_MIMO26_MODEL_LAYER_COUNT
#define SPARK_MIMO26_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAXIMUM_CONTEXT_TOKENS
#define SPARK_MIMO26_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_MIMO26_MODEL_HIDDEN_BF16_BYTES SPARK_MIMO26_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES
#define SPARK_MIMO26_MODEL_LAYER_IS_GDN(layer) 0u

#define SPARK_QWEN38_SERVING_ADAPTER_FN(name) SparkMimo26##name
#define SPARK_QWEN38_SERVING_ADAPTER_TYPE(name) SparkMimo26##name
#define SPARK_QWEN38_SERVING_ADAPTER_CONST(name) SPARK_MIMO26_##name
#define SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION MIMO26_MODEL_REVISION
#define SPARK_QWEN38_SERVING_ADAPTER_CONTRACT_SHA256 MIMO26_CONTRACT_SHA256
#define SPARK_QWEN38_SERVING_ADAPTER_DRIVER_DESCRIPTION_SHA256 MIMO26_MODEL_DESCRIPTION_SHA256
#define SPARK_QWEN38_SERVING_ADAPTER_TP_DEGREE_VALID(tp_degree) \
	((tp_degree) == SPARK_MIMO26_SERVING_PARALLEL_GROUP_SIZE)
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_COUNT(state) \
	(state)->pp_stage_count
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_INDEX(state) \
	SparkMimo26ServingPpStageIndex(state,(state)->stage_index)
#define SPARK_QWEN38_SERVING_ADAPTER_BIND_FAMILY(state) \
	SparkMimo26ServingInitializeFamilyState(state)
#define SPARK_QWEN38_SERVING_ADAPTER_PREFETCH SparkMimo26ServingPrefetch
#define SPARK_QWEN38_SERVING_ADAPTER_RESOLVE_PREFETCH \
	SparkMimo26ServingResolvePrefetch
#define SPARK_QWEN38_SERVING_ADAPTER_RESET SparkMimo26ServingReset
#define SPARK_QWEN38_SERVING_ADAPTER_SUBMISSION_STALE(state,submission) \
	SparkMimo26ServingSubmissionStale((state),(submission))

typedef struct SparkMimo26ServingPending
{
	struct SparkMimo26ServingState *owner;
	SparkServingAdapterPendingCommon common;
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkMimo26ServingPending;

typedef struct SparkMimo26ServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkMimo26ServingTransportShim;

typedef struct SparkMimo26ServingState
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
	uint32_t stage_layer_counts[SPARK_MIMO26_SERVING_PP_STAGE_COUNT];
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
	atomic_uint reset_active;
	atomic_uint_fast64_t reset_generation;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkMimo26KvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkMimo26ServingTransportShim shim;
	SparkMimo26ServingPending pending[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkMimo26ServingState;

static SparkStatus SparkMimo26ServingValidateSubmission(
	void *adapter_state,const SparkModelServingSubmission *submission);
static SparkStatus SparkMimo26ServingQuiesce(
	void *adapter_state,uint64_t deadline_time_ns);

static _Thread_local SparkModelDriverCacheLane SparkMimo26ServingCacheScratch[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

#include "sparkpipe/family/serving/spark_serving_submission_stale.h"

#include "sparkpipe/family/serving/spark_serving_initialize_family_state.h"

#include "sparkpipe/family/serving/spark_serving_cache_context.h"

#include "sparkpipe/family/serving/spark_serving_prefetch.h"

#include "sparkpipe/family/serving/spark_serving_reset_control.h"

#include "sparkpipe/family/serving/spark_serving_reset.h"

static const SparkModelServingAdapterDescriptor SparkMimo26ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_MIMO26_SERVING_ADAPTER_ID,
		SPARK_MIMO26_SERVING_MODEL_ID,
		MIMO26_MODEL_REVISION,
		SPARK_MIMO26_SERVING_PROGRAM_NAME,
		MIMO26_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP),
	.parallel_group_size = SPARK_MIMO26_SERVING_PARALLEL_GROUP_SIZE,
	.stage_count = SPARK_MIMO26_SERVING_STAGE_COUNT,
	.layer_count = SPARK_MIMO26_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_MIMO26_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_MIMO26_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 1u,
	.max_active_sequence_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.stage_layer_counts = {SPARK_MIMO26_SERVING_STAGE_LAYER_LIST},
	.minimum_efficient_submission_row_count = 0u,
	.cache_block_token_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS
};

#include "sparkpipe/spark_qwen38_pp_serving_adapter_common.h"
