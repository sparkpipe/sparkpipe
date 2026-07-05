#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_MAGIC_BYTES 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_MAGIC "SPARKGLM52FP8\0\0"
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_HEADER_BYTES 512u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_ALIGNMENT 4096u

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_WEIGHT 0u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W1_SCALE_INV 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_WEIGHT 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_W2_SCALE_INV 3u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_COUNT 4u

typedef struct SparkGlm52ResidentDecodeStageFp8MoePackRegion
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm52ResidentDecodeStageFp8MoePackRegion;

typedef struct SparkGlm52ResidentDecodeStageFp8MoePackHeader
{
	uint8_t magic[SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_MAGIC_BYTES];
	uint32_t fields[16];
	SparkGlm52ResidentDecodeStageFp8MoePackRegion regions[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PACK_REGION_COUNT];
} SparkGlm52ResidentDecodeStageFp8MoePackHeader;

typedef struct SparkGlm52ResidentDecodeStageFp8MoeResidentBinding
{
	uint32_t abi_version;
	uint32_t layer_index;
	SparkGlm52ResidentDecodeStageFp8MoePlan plan;
	uint8_t *w1_weight_fp8_e4m3;
	float *w1_scale_inv_f32;
	uint8_t *w2_weight_fp8_e4m3;
	float *w2_scale_inv_f32;
	void *workspace;
} SparkGlm52ResidentDecodeStageFp8MoeResidentBinding;

typedef struct SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateInfo
{
	uint32_t abi_version;
	uint32_t layer_index;
	uint32_t maximum_active_sequence_count;
	uint32_t reserved;
	const char *pack_path;
} SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateInfo;

SparkStatus SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateFromPackFile(
	SparkGlm52ResidentDecodeStageFp8MoeResidentBinding *binding,
	const SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateInfo *create_info);

void SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
	SparkGlm52ResidentDecodeStageFp8MoeResidentBinding *binding);

#ifdef __cplusplus
}
#endif
