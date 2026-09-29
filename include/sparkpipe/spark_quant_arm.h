#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_QUANT_ARM_FORMAT "sparkpipe-quant-arm-v1"
#define SPARK_QUANT_ARM_MAX_RANKS 64u
#define SPARK_QUANT_ARM_TEXT_BYTES 257u
#define SPARK_QUANT_ARM_TOKEN_BYTES 33u
#define SPARK_QUANT_ARM_ID_BYTES 257u
#define SPARK_QUANT_ARM_MAX_FILE_BYTES 262144u
#define SPARK_QUANT_ARM_CANONICAL_BYTES 32768u
#define SPARK_QUANT_ARM_KV_TEXT_BYTES 96u

typedef struct SparkQuantArm
{
	char arm_id[SPARK_QUANT_ARM_ID_BYTES];
	char model[SPARK_QUANT_ARM_TOKEN_BYTES];
	char revision[SPARK_QUANT_ARM_TEXT_BYTES];
	uint32_t tensor_parallel;
	uint32_t pipeline_parallel;
	uint32_t kv_shard;
	uint32_t rank_count;
	char spine_frame[SPARK_QUANT_ARM_TOKEN_BYTES];
	char spine_source[SPARK_QUANT_ARM_TEXT_BYTES];
	char spine_digest[SPARK_QUANT_ARM_MAX_RANKS][SPARK_SHA256_HEX_BYTES];
	char expert_codec[SPARK_QUANT_ARM_TOKEN_BYTES];
	char expert_label[SPARK_QUANT_ARM_TOKEN_BYTES];
	char expert_producer[SPARK_QUANT_ARM_TOKEN_BYTES];
	char expert_source[SPARK_QUANT_ARM_TEXT_BYTES];
	uint32_t expert_recipe_present;
	char expert_recipe_sha256[SPARK_SHA256_HEX_BYTES];
	char kv_latent[SPARK_QUANT_ARM_TOKEN_BYTES];
	char kv_index[SPARK_QUANT_ARM_TOKEN_BYTES];
	char kv_state[SPARK_QUANT_ARM_TOKEN_BYTES];
	uint32_t kv_group;
	char kv_mode[SPARK_QUANT_ARM_TOKEN_BYTES];
	char drafter_kind[SPARK_QUANT_ARM_TOKEN_BYTES];
	char drafter_label[SPARK_QUANT_ARM_TOKEN_BYTES];
	char drafter_codec[SPARK_QUANT_ARM_TOKEN_BYTES];
	char drafter_head[SPARK_QUANT_ARM_TOKEN_BYTES];
	uint32_t drafter_sidecar_count;
	char drafter_sidecar_sha256[SPARK_QUANT_ARM_MAX_RANKS][SPARK_SHA256_HEX_BYTES];
	char pack_sha256[SPARK_QUANT_ARM_MAX_RANKS][SPARK_SHA256_HEX_BYTES];
	char module_archive_sha256[SPARK_SHA256_HEX_BYTES];
	char driver_sha256[SPARK_SHA256_HEX_BYTES];
	char adapter_sha256[SPARK_SHA256_HEX_BYTES];
	char arm_digest[SPARK_SHA256_HEX_BYTES];
	char pack_set_sha256[SPARK_SHA256_HEX_BYTES];
	char kv_text[SPARK_QUANT_ARM_KV_TEXT_BYTES];
} SparkQuantArm;

void SparkQuantArmReset(SparkQuantArm *arm);
SparkStatus SparkQuantArmParseText(const char *text, size_t text_bytes, SparkQuantArm *arm, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkQuantArmLoadFile(const char *path, SparkQuantArm *arm, char *error_buffer, uint32_t error_buffer_bytes);
SparkStatus SparkQuantArmCanonicalize(const SparkQuantArm *arm, char *buffer, uint32_t buffer_bytes, uint32_t *written_bytes);
SparkStatus SparkQuantArmExpectedId(const SparkQuantArm *arm, char *buffer, uint32_t buffer_bytes);

#ifdef __cplusplus
}
#endif
