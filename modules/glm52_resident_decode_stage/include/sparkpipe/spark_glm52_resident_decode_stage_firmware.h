#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"

// The batch-variant tuning header: the active-sequence ceiling below IS the
// compiled bucket, so a variant build's lane tables shrink to the tight fit.
// The unflagged build is b1024, the ceiling this stage has always had.
#include "sparkpipe/spark_glm52_batch_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 3u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION 1u
/*
 * TP8 geometry: every rank owns the full 78-layer model as one stage.
 * The old PP13 split (13 stages x 6 layers) is retired.
 */
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE 78u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_BATCH_BUCKET
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT 65536u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT \
	(2u * SPARK_GLM52_MODEL_HIDDEN_DIMENSION)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW \
	(SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * (uint32_t)sizeof(uint32_t))

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL UINT32_C(0x00000001)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT UINT32_C(0x00000002)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT UINT32_C(0x00000004)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT UINT32_C(0x00000008)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT UINT32_C(0x00000010)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS \
	(SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL | \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT | \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT | \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT | \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT)

typedef struct SparkGlm52ResidentDecodeStageNodeContext
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
	/* TP8 hidden-state collectives. tp_degree == 1 leaves every field
	 * unset and the module runs the eager chunk chain without a backend. */
	uint32_t tp_collective_backend_kind;
	uint64_t tp_collective_identifier;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t tp_collective_control_port_base;
	SparkTpDeviceCollectiveTopology tp_collective_topology;
	const char *tp_collective_backend_module_path;
} SparkGlm52ResidentDecodeStageNodeContext;

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES \
	((uint32_t)sizeof(SparkGlm52ResidentDecodeStageNodeContext))

typedef struct SparkGlm52ResidentDecodeStageBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t active_sequence_count;
	const uint32_t *token_ids;
	const uint32_t *row_resident_slots;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkGlm52ResidentDecodeStageBatchView;

typedef struct SparkGlm52ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
	const void *sideband_input;
	uint64_t sideband_input_bytes;
	void *sideband_output;
	uint64_t sideband_output_bytes;
} SparkGlm52ResidentDecodeStageFrameContext;

static inline uint32_t SparkGlm52ResidentDecodeStageFirstLayer(uint32_t stage_index)
{
	return(stage_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE);
}

static inline uint32_t SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(uint32_t source_stage_index)
{
	return(source_stage_index + 1u < SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT && (source_stage_index & 1u) != 0u ? 1u : 0u);
}

static inline uint32_t SparkGlm52ResidentDecodeStageRequiresSidebandInput(uint32_t stage_index)
{
	return(stage_index > 0u ? SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(stage_index - 1u) : 0u);
}

static inline uint32_t SparkGlm52ResidentDecodeStageRequiresSidebandOutput(uint32_t stage_index)
{
	return(SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(stage_index));
}

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);
SparkStatus SparkGlm52ResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame);
SparkStatus SparkGlm52ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot);
void SparkGlm52ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
