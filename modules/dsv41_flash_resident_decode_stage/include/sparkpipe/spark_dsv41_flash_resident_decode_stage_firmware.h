#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv41_flash_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weight_codec.h"

#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS 0u
#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS 0u
#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT 65536u
#define SPARK_DSV41_FLASH_STAGEPACK_MAX_TENSOR_COUNT 8192u

typedef struct SparkDsv41FlashResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t expert_weight_codec;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t execution_row_capacity;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t flags;
	const char *stage_pack_path;
	const char *model_revision;
} SparkDsv41FlashResidentDecodeStageNodeContext;

typedef struct SparkDsv41FlashResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t row_count;
} SparkDsv41FlashResidentDecodeStageFrameContext;

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkDsv41FlashCudaContextEnsure(void);

#ifdef __cplusplus
}
#endif
