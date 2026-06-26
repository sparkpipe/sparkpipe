#include "sparkpipe/spark_glm_final_logits.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static uint64_t SparkGlmFinalLogitsMixFloat(uint64_t checksum, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return SparkMixU64(checksum, bits);
}

static SparkStatus SparkGlmFinalLogitsBuildHidden(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, float *hidden_values)
{
    uint32_t hidden_index;
    float sum_squares;
    float inverse_rms;

    if (request == 0 || activation_bf16 == 0 || hidden_values == 0 || activation_count < request->hidden_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!request->activation_already_normed && (norm_weight_bf16 == 0 || norm_weight_count < request->hidden_size))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    sum_squares = 0.0f;
    for (hidden_index = 0u; hidden_index < request->hidden_size; ++hidden_index)
    {
        float value;

        value = SparkGlmBf16ToFloat(activation_bf16[hidden_index]);
        if (!isfinite(value))
            value = 0.0f;
        hidden_values[hidden_index] = value;
        sum_squares += value * value;
    }
    inverse_rms = request->activation_already_normed ? 1.0f : (1.0f / sqrtf((sum_squares / (float)request->hidden_size) + request->rms_norm_epsilon));
    if (!isfinite(inverse_rms))
        inverse_rms = 1.0f;
    for (hidden_index = 0u; hidden_index < request->hidden_size; ++hidden_index)
    {
        float weight;

        weight = request->activation_already_normed ? 1.0f : SparkGlmBf16ToFloat(norm_weight_bf16[hidden_index]);
        if (!isfinite(weight))
            weight = 0.0f;
        hidden_values[hidden_index] = hidden_values[hidden_index] * inverse_rms * weight;
        if (!isfinite(hidden_values[hidden_index]))
            hidden_values[hidden_index] = 0.0f;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlmFinalLogitsResetReport(const SparkGlmFinalLogitsRequest *request, SparkGlmFinalLogitsReport *report)
{
    memset(report, 0, sizeof(*report));
    report->observed_token = 0u;
    report->hidden_size = request->hidden_size;
    report->vocab_size = request->vocab_size;
    report->best_logit = -FLT_MAX;
    report->logits_checksum = 0x5350474C4D4C4843ull;
}

static void SparkGlmFinalLogitsUpdateBest(const SparkGlmFinalLogitsRequest *request, SparkGlmFinalLogitsReport *report, const float *hidden_values, const uint16_t *head_row_bf16, uint32_t token_index)
{
    uint32_t hidden_index;
    float logit;

    logit = 0.0f;
    for (hidden_index = 0u; hidden_index < request->hidden_size; ++hidden_index)
    {
        float weight;

        weight = SparkGlmBf16ToFloat(head_row_bf16[hidden_index]);
        if (!isfinite(weight))
            weight = 0.0f;
        logit += hidden_values[hidden_index] * weight;
    }
    if (!isfinite(logit))
    {
        logit = 0.0f;
    }
    report->logits_checksum = SparkGlmFinalLogitsMixFloat(report->logits_checksum, logit);
    if (logit > report->best_logit || (logit == report->best_logit && token_index < report->observed_token))
    {
        report->best_logit = logit;
        report->observed_token = token_index;
    }
}

static void SparkGlmFinalLogitsFinishReport(const SparkGlmFinalLogitsRequest *request, SparkGlmFinalLogitsReport *report)
{
    uint64_t checksum;

    checksum = 0x5350474C4D4C5452ull;
    checksum = SparkMixU64(checksum, request->hidden_size);
    checksum = SparkMixU64(checksum, request->vocab_size);
    checksum = SparkMixU64(checksum, request->activation_already_normed ? 1u : 0u);
    checksum = SparkMixU64(checksum, report->activation_checksum);
    checksum = SparkMixU64(checksum, report->norm_checksum);
    checksum = SparkMixU64(checksum, report->head_checksum);
    checksum = SparkMixU64(checksum, report->logits_checksum);
    checksum = SparkMixU64(checksum, report->observed_token);
    checksum = SparkGlmFinalLogitsMixFloat(checksum, report->best_logit);
    report->trace_checksum = checksum;
}

float SparkGlmBf16ToFloat(uint16_t value)
{
    uint32_t bits;
    float result;

    bits = ((uint32_t)value) << 16u;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t SparkGlmFloatToBf16(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16u);
}

uint64_t SparkComputeGlmBf16HostChecksum(const uint16_t *values, uint64_t value_count)
{
    uint64_t value_index;
    uint64_t checksum;

    if (values == 0 || value_count == 0u)
    {
        return 0u;
    }
    checksum = 0x5350474C4D424631ull;
    checksum = SparkMixU64(checksum, value_count);
    for (value_index = 0u; value_index < value_count; ++value_index)
    {
        checksum = SparkMixU64(checksum, values[value_index]);
    }
    return checksum;
}

void SparkGlmFinalLogitsRequestReset(SparkGlmFinalLogitsRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->rms_norm_epsilon = 1.0e-6f;
    request->sentinel = SPARKPIPE_GLM_FINAL_LOGITS_SENTINEL;
}

SparkStatus SparkValidateGlmFinalLogitsRequest(const SparkGlmFinalLogitsRequest *request)
{
    if (request == 0 || request->sentinel != SPARKPIPE_GLM_FINAL_LOGITS_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->hidden_size == 0u || request->hidden_size > SPARKPIPE_DEFAULT_HIDDEN_SIZE || request->vocab_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!request->activation_already_normed && (!isfinite(request->rms_norm_epsilon) || request->rms_norm_epsilon <= 0.0f))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRunGlmFinalLogitsHostReference(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const uint16_t *head_weight_bf16, uint64_t head_weight_count, SparkGlmFinalLogitsReport *report)
{
    float hidden_values[SPARKPIPE_DEFAULT_HIDDEN_SIZE];
    uint32_t token_index;
    SparkStatus status;

    if (report == 0 || head_weight_bf16 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkValidateGlmFinalLogitsRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (head_weight_count < (uint64_t)request->hidden_size * (uint64_t)request->vocab_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlmFinalLogitsBuildHidden(request, activation_bf16, activation_count, norm_weight_bf16, norm_weight_count, hidden_values);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlmFinalLogitsResetReport(request, report);
    report->activation_checksum = SparkComputeGlmBf16HostChecksum(activation_bf16, request->hidden_size);
    report->norm_checksum = request->activation_already_normed ? 0u : SparkComputeGlmBf16HostChecksum(norm_weight_bf16, request->hidden_size);
    report->head_checksum = SparkComputeGlmBf16HostChecksum(head_weight_bf16, (uint64_t)request->hidden_size * (uint64_t)request->vocab_size);
    for (token_index = 0u; token_index < request->vocab_size; ++token_index)
    {
        SparkGlmFinalLogitsUpdateBest(request, report, hidden_values, head_weight_bf16 + ((uint64_t)token_index * (uint64_t)request->hidden_size), token_index);
    }
    report->row_count = request->vocab_size;
    SparkGlmFinalLogitsFinishReport(request, report);
    return report->nonfinite_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

SparkStatus SparkRunGlmFinalLogitsFileHostReference(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const char *head_weight_path, uint64_t head_weight_offset_bytes, uint64_t head_weight_bytes, SparkGlmFinalLogitsReport *report)
{
    float hidden_values[SPARKPIPE_DEFAULT_HIDDEN_SIZE];
    uint16_t row_values[SPARKPIPE_DEFAULT_HIDDEN_SIZE];
    FILE *file;
    uint32_t token_index;
    SparkStatus status;

    if (report == 0 || head_weight_path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkValidateGlmFinalLogitsRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (head_weight_bytes < (uint64_t)request->hidden_size * (uint64_t)request->vocab_size * sizeof(uint16_t))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (head_weight_offset_bytes > (uint64_t)LONG_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlmFinalLogitsBuildHidden(request, activation_bf16, activation_count, norm_weight_bf16, norm_weight_count, hidden_values);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    file = fopen(head_weight_path, "rb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fseek(file, (long)head_weight_offset_bytes, SEEK_SET) != 0)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    SparkGlmFinalLogitsResetReport(request, report);
    report->activation_checksum = SparkComputeGlmBf16HostChecksum(activation_bf16, request->hidden_size);
    report->norm_checksum = request->activation_already_normed ? 0u : SparkComputeGlmBf16HostChecksum(norm_weight_bf16, request->hidden_size);
    for (token_index = 0u; token_index < request->vocab_size; ++token_index)
    {
        if (fread(row_values, sizeof(uint16_t), request->hidden_size, file) != request->hidden_size)
        {
            fclose(file);
            return SPARK_STATUS_IO_ERROR;
        }
        report->head_checksum = SparkMixU64(report->head_checksum, SparkComputeGlmBf16HostChecksum(row_values, request->hidden_size));
        SparkGlmFinalLogitsUpdateBest(request, report, hidden_values, row_values, token_index);
    }
    fclose(file);
    report->row_count = request->vocab_size;
    SparkGlmFinalLogitsFinishReport(request, report);
    return report->nonfinite_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static bool SparkGlmFinalLogitsAllowedTokensAreValid(const SparkGlmFinalLogitsRequest *request, const uint32_t *allowed_token_ids, uint32_t allowed_token_count)
{
    uint32_t allowed_index;

    if (request == 0 || allowed_token_ids == 0 || allowed_token_count == 0u || allowed_token_count > SPARKPIPE_MAX_CONSTRAINT_TOKENS)
    {
        return false;
    }

    for (allowed_index = 0u; allowed_index < allowed_token_count; ++allowed_index)
    {
        if (allowed_token_ids[allowed_index] >= request->vocab_size)
        {
            return false;
        }
    }

    return true;
}

SparkStatus SparkRunGlmFinalLogitsFileHostReferenceRestricted(const SparkGlmFinalLogitsRequest *request, const uint16_t *activation_bf16, uint32_t activation_count, const uint16_t *norm_weight_bf16, uint32_t norm_weight_count, const char *head_weight_path, uint64_t head_weight_offset_bytes, uint64_t head_weight_bytes, const uint32_t *allowed_token_ids, uint32_t allowed_token_count, SparkGlmFinalLogitsReport *report)
{
    float hidden_values[SPARKPIPE_DEFAULT_HIDDEN_SIZE];
    uint16_t row_values[SPARKPIPE_DEFAULT_HIDDEN_SIZE];
    FILE *file;
    uint32_t allowed_index;
    SparkStatus status;

    if (report == 0 || head_weight_path == 0 || allowed_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkValidateGlmFinalLogitsRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkGlmFinalLogitsAllowedTokensAreValid(request, allowed_token_ids, allowed_token_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (head_weight_bytes < (uint64_t)request->hidden_size * (uint64_t)request->vocab_size * sizeof(uint16_t))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (head_weight_offset_bytes > (uint64_t)LONG_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlmFinalLogitsBuildHidden(request, activation_bf16, activation_count, norm_weight_bf16, norm_weight_count, hidden_values);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    file = fopen(head_weight_path, "rb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    SparkGlmFinalLogitsResetReport(request, report);
    report->activation_checksum = SparkComputeGlmBf16HostChecksum(activation_bf16, request->hidden_size);
    report->norm_checksum = request->activation_already_normed ? 0u : SparkComputeGlmBf16HostChecksum(norm_weight_bf16, request->hidden_size);

    for (allowed_index = 0u; allowed_index < allowed_token_count; ++allowed_index)
    {
        uint32_t token_index;
        uint64_t row_offset_bytes;

        token_index = allowed_token_ids[allowed_index];
        row_offset_bytes = head_weight_offset_bytes + ((uint64_t)token_index * (uint64_t)request->hidden_size * sizeof(uint16_t));
        if (row_offset_bytes > (uint64_t)LONG_MAX)
        {
            fclose(file);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (fseek(file, (long)row_offset_bytes, SEEK_SET) != 0)
        {
            fclose(file);
            return SPARK_STATUS_IO_ERROR;
        }
        if (fread(row_values, sizeof(uint16_t), request->hidden_size, file) != request->hidden_size)
        {
            fclose(file);
            return SPARK_STATUS_IO_ERROR;
        }
        report->head_checksum = SparkMixU64(report->head_checksum, SparkComputeGlmBf16HostChecksum(row_values, request->hidden_size));
        SparkGlmFinalLogitsUpdateBest(request, report, hidden_values, row_values, token_index);
    }

    fclose(file);
    report->row_count = allowed_token_count;
    SparkGlmFinalLogitsFinishReport(request, report);
    return report->nonfinite_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}
