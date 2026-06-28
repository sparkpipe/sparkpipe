#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_GLM_FINAL_LOGITS_SENTINEL 0x5350474C4D4C4F47ull

typedef struct SparkGlmFinalLogitsRequest
{
    uint32_t hidden_size;
    uint32_t vocab_size;
    bool activation_already_normed;
    float rms_norm_epsilon;
    uint64_t sentinel;
} SparkGlmFinalLogitsRequest;

typedef struct SparkGlmFinalLogitsReport
{
    uint32_t observed_token;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t row_count;
    uint32_t nonfinite_count;
    uint64_t activation_checksum;
    uint64_t norm_checksum;
    uint64_t head_checksum;
    uint64_t logits_checksum;
    uint64_t trace_checksum;
    float best_logit;
} SparkGlmFinalLogitsReport;

float SparkGlmBf16ToFloat(uint16_t value);
uint16_t SparkGlmFloatToBf16(float value);
uint64_t SparkComputeGlmBf16HostChecksum(const uint16_t *values, uint64_t value_count);
void SparkGlmFinalLogitsRequestReset(SparkGlmFinalLogitsRequest *request);
SparkStatus SparkValidateGlmFinalLogitsRequest(const SparkGlmFinalLogitsRequest *request);
SparkStatus SparkRunGlmFinalLogitsHostReference(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const uint16_t *head_weight_bf16, uint64_t head_weight_count, SparkGlmFinalLogitsReport *report);
SparkStatus SparkRunGlmFinalLogitsFileHostReference(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const char *head_weight_path, uint64_t head_weight_offset_bytes, uint64_t head_weight_bytes, SparkGlmFinalLogitsReport *report);
SparkStatus SparkRunGlmFinalLogitsFileHostReferenceRestricted(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const char *head_weight_path, uint64_t head_weight_offset_bytes, uint64_t head_weight_bytes, const uint32_t *allowed_token_ids, uint32_t allowed_token_count, SparkGlmFinalLogitsReport *report);

#ifdef __cplusplus
}
#endif
