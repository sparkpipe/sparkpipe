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

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 4u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION 1u
/*
 * Supported resident geometries. The default deployment is TP8: one firmware
 * stage owns all 78 layers. The pipeline alternative is the model's PP7
 * split [12,11x6], whose per-stage table is generated into
 * spark_glm52_model.h from the contract (single source for the packer, the
 * module configure chain and the DSpark dispatch policy). No uniform
 * layers-per-stage multiply exists anywhere: first layers are prefix sums.
 */
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_LAYERS_PER_STAGE \
	SPARK_GLM52_MODEL_LAYER_COUNT
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
	/* JIT-KV page-store host backing (flows from the serving adapter's
	 * kv_backing_directory / kv_backing_maximum_bytes). */
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
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

typedef struct SparkGlm52ResidentDecodeStageGeometry
{
	uint32_t stage_count;
	const uint32_t *stage_first_layers;
	const uint32_t *stage_layer_counts;
} SparkGlm52ResidentDecodeStageGeometry;

static const uint32_t SparkGlm52ResidentDecodeStageTp8FirstLayers[1u] = { 0u };
static const uint32_t SparkGlm52ResidentDecodeStageTp8LayerCounts[1u] =
	{ SPARK_GLM52_MODEL_LAYER_COUNT };
static const SparkGlm52ResidentDecodeStageGeometry
	SparkGlm52ResidentDecodeStageTp8Geometry =
	{
		SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,
		SparkGlm52ResidentDecodeStageTp8FirstLayers,
		SparkGlm52ResidentDecodeStageTp8LayerCounts
	};

static const uint32_t SparkGlm52ResidentDecodeStagePp7FirstLayers[
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_FIRST_LAYER_INITIALIZER;
static const uint32_t SparkGlm52ResidentDecodeStagePp7LayerCounts[
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT] =
	SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNTS_INITIALIZER;
static const SparkGlm52ResidentDecodeStageGeometry
	SparkGlm52ResidentDecodeStagePp7Geometry =
	{
		SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT,
		SparkGlm52ResidentDecodeStagePp7FirstLayers,
		SparkGlm52ResidentDecodeStagePp7LayerCounts
	};

/* Resolve the geometry a node-context stage_count claims; 0 when unknown. */
static inline const SparkGlm52ResidentDecodeStageGeometry *
SparkGlm52ResidentDecodeStageGeometryFor(uint32_t stage_count)
{
	if ( stage_count == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT )
		return(&SparkGlm52ResidentDecodeStageTp8Geometry);
	if ( stage_count == SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT )
		return(&SparkGlm52ResidentDecodeStagePp7Geometry);
	return(0);
}

static inline uint32_t SparkGlm52ResidentDecodeStageFirstLayer(
	const SparkGlm52ResidentDecodeStageGeometry *geometry,uint32_t stage_index)
{
	return(geometry == 0u || stage_index >= geometry->stage_count ?
		0u : geometry->stage_first_layers[stage_index]);
}

typedef struct SparkGlm52ResidentDecodeStageDsaTap
{
	uint32_t layer_index;
	uint32_t full_indexer;
} SparkGlm52ResidentDecodeStageDsaTap;

/*
 * The five aux capture taps {7,22,38,54,69} (aux ids minus the capture
 * offset) with their TRUE indexer coverage derived from the shared rule —
 * tests/test_glm52_stagepack.c pins these literals against the aux id table.
 * Under the PP7 split the parity heuristic claimed carries from stages
 * {1,3,5} while only stages whose LAST layer has full indexer coverage
 * ({1,6}: layers 22 and 66) actually bound one; the per-stage derivation
 * below replaces the heuristic outright.
 */
static const SparkGlm52ResidentDecodeStageDsaTap
	SparkGlm52ResidentDecodeStageDsaTaps[] =
	{
		{ 7u, SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(7u) },
		{ 22u, SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(22u) },
		{ 38u, SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(38u) },
		{ 54u, SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(54u) },
		{ 69u, SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(69u) }
	};

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_TAP_COUNT \
	((uint32_t)(sizeof(SparkGlm52ResidentDecodeStageDsaTaps) / \
		sizeof(SparkGlm52ResidentDecodeStageDsaTaps[0])))

/* Source stage S carries the DSA sideband iff its LAST layer has full
 * indexer coverage — a statement about indexer placement, not position. */
static inline uint32_t SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(
	const SparkGlm52ResidentDecodeStageGeometry *geometry,
	uint32_t source_stage_index)
{
	uint32_t last_layer;
	if ( geometry == 0u || source_stage_index + 1u >= geometry->stage_count )
		return(0u);
	last_layer = geometry->stage_first_layers[source_stage_index] +
		geometry->stage_layer_counts[source_stage_index] - 1u;
	return(SPARK_GLM52_MODEL_LAYER_HAS_FULL_INDEXER(last_layer));
}

static inline uint32_t SparkGlm52ResidentDecodeStageRequiresSidebandInput(
	const SparkGlm52ResidentDecodeStageGeometry *geometry,
	uint32_t stage_index)
{
	return(stage_index > 0u ?
		SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(geometry,stage_index - 1u) : 0u);
}

static inline uint32_t SparkGlm52ResidentDecodeStageRequiresSidebandOutput(
	const SparkGlm52ResidentDecodeStageGeometry *geometry,
	uint32_t stage_index)
{
	return(SparkGlm52ResidentDecodeStageBoundaryCarriesDsa(geometry,stage_index));
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
