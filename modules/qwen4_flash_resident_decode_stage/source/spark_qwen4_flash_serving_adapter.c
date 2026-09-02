
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_qwen4_flash_model.h"
#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_qwen4_flash_serving_adapter.h"
#include "sparkpipe/spark_serving_adapter_template.h"

#ifndef QWEN4_FLASH_MODEL_REVISION
#error "QWEN4_FLASH_MODEL_REVISION must name the exact source snapshot revision"
#endif
#ifndef QWEN4_FLASH_CONTRACT_SHA256
#error "QWEN4_FLASH_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_QWEN4_FLASH_SERVING_ADAPTER_ID \
	"spark.qwen4_flash.serving-adapter.tp4pp4.v1"
#define SPARK_QWEN4_FLASH_SERVING_MODEL_ID "Qwen/Qwen3.8-Flash-Next"
#define SPARK_QWEN4_FLASH_SERVING_DRIVER_MODEL_ID \
	"qwen4_flash.resident-decode-stage-firmware"
#define SPARK_QWEN4_FLASH_SERVING_STAGE_NAME "qwen4_flash_resident_decode_stage"
#define SPARK_QWEN4_FLASH_SERVING_TARGET \
	"cuda.sm121.qwen4_flash.resident_decode_stage.fp8"
#define SPARK_QWEN4_FLASH_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_QWEN4_FLASH_SERVING_STAGE_COUNT 16u
#define SPARK_QWEN4_FLASH_SERVING_DEFAULT_TP_DEGREE 4u
#define SPARK_QWEN4_FLASH_SERVING_PARALLEL_GROUP_SIZE 4u
#define SPARK_QWEN4_FLASH_SERVING_PP_STAGE_COUNT \
	(SPARK_QWEN4_FLASH_SERVING_STAGE_COUNT / SPARK_QWEN4_FLASH_SERVING_PARALLEL_GROUP_SIZE)
#define SPARK_QWEN4_FLASH_SERVING_MAX_PP_STAGE_COUNT \
	SPARK_QWEN4_FLASH_SERVING_PP_STAGE_COUNT
#define SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT \
	(SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT / SPARK_QWEN4_FLASH_SERVING_PP_STAGE_COUNT)
#define SPARK_QWEN4_FLASH_SERVING_MAX_SEQUENCE_POSITIONS_CAP \
	SPARK_QWEN4_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_QWEN4_FLASH_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_QWEN38_SERVING_ADAPTER_FN(name) SparkQwen4Flash##name
#define SPARK_QWEN38_SERVING_ADAPTER_TYPE(name) SparkQwen4Flash##name
#define SPARK_QWEN38_SERVING_ADAPTER_CONST(name) SPARK_QWEN4_FLASH_##name
#define SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION QWEN4_FLASH_MODEL_REVISION
#define SPARK_QWEN38_SERVING_ADAPTER_CONTRACT_SHA256 QWEN4_FLASH_CONTRACT_SHA256
#define SPARK_QWEN38_SERVING_ADAPTER_TP_DEGREE_VALID(tp_degree) \
	((tp_degree) == SPARK_QWEN4_FLASH_SERVING_PARALLEL_GROUP_SIZE)
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_COUNT(state) \
	(state)->pp_stage_count
#define SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_INDEX(state) \
	SparkQwen4FlashServingPpStageIndex(state,(state)->stage_index)
#define SPARK_QWEN38_SERVING_ADAPTER_BIND_FAMILY(state) SPARK_STATUS_OK

typedef struct SparkQwen4FlashServingPending
{
	struct SparkQwen4FlashServingState *owner;
	SparkServingAdapterPendingCommon common;
	uint64_t frame_sequence_id;
	uint64_t frame_sequence_position;
	SparkStatus frame_status;
	SparkModelDriverResidencyToken residency;
	uint64_t accepted_token_count;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint32_t last_row_by_lane[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_slots[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_row_flats[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t output_token_ids[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_output_ids[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t frame_token_ids[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkQwen4FlashServingPending;

typedef struct SparkQwen4FlashServingTransportShim
{
	const void *input_base;
	const uint32_t *input_row_map;
	uint32_t input_rows;
	void *input_scratch;
	void *output_base;
	const uint32_t *output_row_map;
	void *execution_stream;
} SparkQwen4FlashServingTransportShim;

typedef struct SparkQwen4FlashServingState
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
	uint32_t stage_layer_counts[SPARK_QWEN4_FLASH_SERVING_PP_STAGE_COUNT];
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
	SparkQwen4FlashKvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t lane_block_counts[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_context_tokens[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	void *gather_scratch;
	SparkQwen4FlashServingTransportShim shim;
	SparkQwen4FlashServingPending pending[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkQwen4FlashServingState;

static const SparkModelServingAdapterDescriptor SparkQwen4FlashServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_QWEN4_FLASH_SERVING_ADAPTER_ID,
		SPARK_QWEN4_FLASH_SERVING_MODEL_ID,
		QWEN4_FLASH_MODEL_REVISION,
		SPARK_QWEN4_FLASH_SERVING_PROGRAM_NAME,
		QWEN4_FLASH_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP),
	.parallel_group_size = SPARK_QWEN4_FLASH_SERVING_PARALLEL_GROUP_SIZE,
	.stage_count = SPARK_QWEN4_FLASH_SERVING_STAGE_COUNT,
	.layer_count = SPARK_QWEN4_FLASH_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,
	.boundary_element_bytes = SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 1u,
	.max_active_sequence_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_resident_sequence_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.stage_layer_counts = {SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT,SPARK_QWEN4_FLASH_SERVING_STAGE_LAYER_COUNT},
	.minimum_efficient_submission_row_count = 0u
};

#include "sparkpipe/spark_qwen38_pp_serving_adapter_common.h"
