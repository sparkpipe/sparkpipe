#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"

#include "sparkpipe/spark_glm5_next_batch_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 5u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION 1u
#ifndef SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_STAGE_COUNT
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_STAGE_COUNT 1u
#endif
#ifndef SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE \
	SPARK_GLM5_NEXT_MODEL_LAYER_COUNT
#endif
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_BATCH_BUCKET
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT 65536u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT \
	SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES 2u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND 1u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW \
	(SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH * (uint32_t)sizeof(uint32_t))

#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL UINT32_C(0x00000001)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT UINT32_C(0x00000002)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT UINT32_C(0x00000004)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT UINT32_C(0x00000008)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT UINT32_C(0x00000010)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS \
	(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL | \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT | \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT | \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT | \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT)

#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTITIONS 16u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS \
	(SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION + 2u)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,heads) \
	((uint64_t)(rows) * (heads) * \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTITIONS)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(rows,heads) \
	(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,heads) * \
	 SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS * \
	 (uint64_t)sizeof(float))

typedef struct SparkGlm5NextResidentDecodeStageNodeContext
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
	const char *stage_pack_path;
	const char *model_revision;
	uint32_t tp_collective_backend_kind;
	uint64_t tp_collective_identifier;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t tp_collective_control_port_base;
	SparkTpDeviceCollectiveTopology tp_collective_topology;
	const char *tp_collective_backend_module_path;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	uint32_t decode_split_context_threshold;
} SparkGlm5NextResidentDecodeStageNodeContext;

#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES \
	((uint32_t)sizeof(SparkGlm5NextResidentDecodeStageNodeContext))

typedef struct SparkGlm5NextResidentDecodeStageBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t active_sequence_count;
	const uint32_t *token_ids;
	const uint32_t *row_resident_slots;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkGlm5NextResidentDecodeStageBatchView;

typedef struct SparkGlm5NextResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
	const void *sideband_input;
	uint64_t sideband_input_bytes;
	void *sideband_output;
	uint64_t sideband_output_bytes;
} SparkGlm5NextResidentDecodeStageFrameContext;

static inline uint32_t SparkGlm5NextResidentDecodeStageFirstLayer(uint32_t stage_index)
{
	return(stage_index * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE);
}

static inline uint32_t SparkGlm5NextResidentDecodeStageBoundaryCarriesDsa(uint32_t source_stage_index)
{
	return(source_stage_index + 1u < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_STAGE_COUNT && (source_stage_index & 1u) != 0u ? 1u : 0u);
}

static inline uint32_t SparkGlm5NextResidentDecodeStageRequiresSidebandInput(uint32_t stage_index)
{
	return(stage_index > 0u ? SparkGlm5NextResidentDecodeStageBoundaryCarriesDsa(stage_index - 1u) : 0u);
}

static inline uint32_t SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(uint32_t stage_index)
{
	return(SparkGlm5NextResidentDecodeStageBoundaryCarriesDsa(stage_index));
}

SparkStatus SparkGlm5NextResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);
SparkStatus SparkGlm5NextResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame);
SparkStatus SparkGlm5NextResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkGlm5NextResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot);
void SparkGlm5NextResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
