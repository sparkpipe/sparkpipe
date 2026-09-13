#pragma once

#include <stdint.h>

#include "sparkpipe/spark_laguna_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"

#include "sparkpipe/spark_laguna_batch_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 7u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 2u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS UINT32_C(0)
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYER_COUNT \
	SPARK_LAGUNA_MODEL_LAYER_COUNT
#ifndef SPARK_LAGUNA_RESIDENT_DECODE_STAGE_STAGE_COUNT
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_STAGE_COUNT 1u
#endif
#ifndef SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE \
	SPARK_LAGUNA_MODEL_LAYER_COUNT
#endif
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_BATCH_BUCKET
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT 65536u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT \
	SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES 2u

#define SPARK_FIRMWARE_LAYERS_PER_STAGE \
	SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE
#define SPARK_FIRMWARE_FRAME_KNOWN_FLAGS_MASK \
	(SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL | \
	 SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT | \
	 SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT)
#include "sparkpipe/spark_resident_decode_stage_firmware_common.h"

#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL \
	SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT \
	SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT \
	SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT
#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS \
	SPARK_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS

typedef struct SparkLagunaResidentDecodeStageNodeContext
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
	uint16_t tp_collective_session_ports[
		SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
		[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	const char *tp_collective_backend_module_path;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	uint32_t decode_split_context_threshold;
	uint32_t flags;
} SparkLagunaResidentDecodeStageNodeContext;

#define SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES \
	((uint32_t)sizeof(SparkLagunaResidentDecodeStageNodeContext))

typedef SparkResidentDecodeStageBatchView SparkLagunaResidentDecodeStageBatchView;
typedef SparkResidentDecodeStageFrameContext SparkLagunaResidentDecodeStageFrameContext;

typedef struct SparkLagunaLinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkLagunaLinearView;

static inline uint32_t SparkLagunaResidentDecodeStageSpanIsValid(uint32_t stage_count,uint32_t stage_index,uint32_t first_layer,uint32_t layer_count)
{
	uint32_t end;
	if ( (stage_count != 1u && stage_count != 2u && stage_count != 4u) || stage_index >= stage_count || first_layer >= SPARK_LAGUNA_MODEL_LAYER_COUNT || layer_count == 0u || layer_count > (SPARK_LAGUNA_MODEL_LAYER_COUNT - first_layer) )
		return(0u);
	end = (first_layer + layer_count);
	if ( (stage_index == 0u) != (first_layer == 0u) || ((stage_index + 1u) == stage_count) != (end == SPARK_LAGUNA_MODEL_LAYER_COUNT) )
		return(0u);
	if ( layer_count != SPARK_LAGUNA_MODEL_STAGE_LAYER_COUNT(stage_count,stage_index) )
		return(0u);
	return(1u);
}

SparkStatus SparkLagunaResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);
SparkStatus SparkLagunaResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame);
SparkStatus SparkLagunaResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkLagunaResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot);
void SparkLagunaResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
