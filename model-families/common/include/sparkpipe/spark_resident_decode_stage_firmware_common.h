#pragma once

#include <stdint.h>

#ifndef SPARK_FIRMWARE_LAYERS_PER_STAGE
#error "llm_defines: set SPARK_FIRMWARE_LAYERS_PER_STAGE (the family layer count per stage)"
#endif
#ifndef SPARK_FIRMWARE_FRAME_KNOWN_FLAGS_MASK
#error "llm_defines: set SPARK_FIRMWARE_FRAME_KNOWN_FLAGS_MASK (the family's legal frame flag set)"
#endif

#define SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL UINT32_C(0x00000001)
#define SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT UINT32_C(0x00000002)
#define SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT UINT32_C(0x00000004)
#define SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT UINT32_C(0x00000008)
#define SPARK_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT UINT32_C(0x00000010)

#define SPARK_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS \
	SPARK_FIRMWARE_FRAME_KNOWN_FLAGS_MASK

typedef struct SparkResidentDecodeStageBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t active_sequence_count;
	const uint32_t *token_ids;
	const uint32_t *row_resident_slots;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkResidentDecodeStageBatchView;

typedef struct SparkResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkResidentDecodeStageBatchView *batch;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
} SparkResidentDecodeStageFrameContext;

static inline uint32_t SparkResidentDecodeStageFirstLayer(uint32_t stage_index)
{
	return(stage_index * SPARK_FIRMWARE_LAYERS_PER_STAGE);
}
