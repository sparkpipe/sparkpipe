#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"

// The batch-variant tuning header: the active-sequence ceiling below IS the
// compiled bucket, so a variant build's lane tables shrink to the tight fit.
// The unflagged build is b1024, the ceiling this stage has always had.
#include "sparkpipe/spark_glm5_next_batch_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 6u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP UINT32_C(0x00000001)
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS \
	SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP
#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH 2u
/*
 * Whole-stack geometry: one stage owning all 45 weight layers (TP16 per
 * the fleet topology; the MTP layer 45 rides the spec path, not this
 * count). The stage split is compile-time overridable for the
 * mid-pipeline validation tier (STAGE_COUNT=2 STAGE_LAYER_COUNT=4
 * MTP_LAYER_COUNT=0 builds a small module against a synthesized pack).
 */
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
/* The inter-stage boundary carries ONE hidden row per token: the HC
 * streams expand at load (every stream initialises to the boundary row)
 * and collapse by unweighted mean at store - the boundary contract is a
 * single hidden row, not the stream set. */
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

/* R3 flash-decode partials workspace, sized per execution slot: one block of
 * (max, sum, latent accumulator) floats per (row, head, partition). The
 * partition cap is the module's copy of the shared split policy - layer.cuh
 * static_asserts the two never drift. */
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
	/* TP8 hidden-state collectives. tp_degree == 1 leaves every field
	 * unset and the module runs the eager chunk chain without a backend. */
	uint32_t tp_collective_backend_kind;
	uint64_t tp_collective_identifier;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t tp_collective_control_port_base;
	SparkTpDeviceCollectiveTopology tp_collective_topology;
	const char *tp_collective_backend_module_path;
	/* JIT-KV page-store host backing (flows from the serving adapter's
	 * kv_backing_directory / kv_backing_maximum_bytes). */
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	/* R3 flash-decode: the decode attention splits the position range across
	 * CTAs once the walk reaches this many positions. 0 keeps the
	 * single-pass kernel byte-for-byte at EVERY context - the shipped
	 * default until the GPU cell qualifies the split path. */
	uint32_t decode_split_context_threshold;
	uint32_t flags;
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

typedef struct SparkGlm5NextKdaReplayLayout
{
	uint64_t pre_q_offset;
	uint64_t pre_k_offset;
	uint64_t pre_v_offset;
	uint64_t key_offset;
	uint64_t value_offset;
	uint64_t retention_offset;
	uint64_t write_gate_offset;
	uint64_t layer_bytes;
} SparkGlm5NextKdaReplayLayout;

static inline SparkGlm5NextKdaReplayLayout SparkGlm5NextKdaReplayLayoutFor(
	uint32_t rank_kda_heads,
	uint32_t steps)
{
	SparkGlm5NextKdaReplayLayout layout;
	uint64_t rank_qk_bytes;
	uint64_t rank_value_bytes;
	uint64_t rank_retention_bytes;
	uint64_t rank_gate_bytes;
	uint64_t cursor;
	rank_qk_bytes = (uint64_t)rank_kda_heads *
		SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION *
		SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES * steps;
	rank_value_bytes = (uint64_t)rank_kda_heads *
		SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION *
		SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES * steps;
	rank_retention_bytes = (uint64_t)rank_kda_heads *
		SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * (uint64_t)sizeof(float) * steps;
	rank_gate_bytes = (uint64_t)rank_kda_heads * (uint64_t)sizeof(float) * steps;
	cursor = 0u;
	layout.pre_q_offset = cursor;
	cursor += rank_qk_bytes;
	layout.pre_k_offset = cursor;
	cursor += rank_qk_bytes;
	layout.pre_v_offset = cursor;
	cursor += rank_value_bytes;
	layout.key_offset = cursor;
	cursor += rank_qk_bytes;
	layout.value_offset = cursor;
	cursor += rank_value_bytes;
	layout.retention_offset = cursor;
	cursor += rank_retention_bytes;
	layout.write_gate_offset = cursor;
	cursor += rank_gate_bytes;
	layout.layer_bytes = cursor;
	return(layout);
}

#define SPARK_GLM5_NEXT_MTP_REPLAY_STEP_BYTES 32u

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
