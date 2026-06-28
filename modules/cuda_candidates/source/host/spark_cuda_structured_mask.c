#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_structured_mask.h"

static uint64_t SparkCudaStructuredMaskLogitCount(const SparkCudaStructuredMaskRequest *request)
{
    if (request == 0)
    {
        return 0;
    }

    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

static bool SparkStructuredMaskTokenIsDuplicate(const uint32_t *allowed_tokens, uint32_t start_index, uint32_t token_index, uint32_t token_id)
{
    uint32_t compare_index;

    for (compare_index = start_index; compare_index < token_index; ++compare_index)
    {
        if (allowed_tokens[compare_index] == token_id)
        {
            return true;
        }
    }

    return false;
}

static uint64_t SparkStructuredMaskTraceChecksum(const SparkCudaStructuredMaskRequest *request, const SparkCudaStructuredMaskReport *report)
{
    uint64_t checksum;

    checksum = 0x53504B4D41534B54ull;
    checksum = SparkMixU64(checksum, request->row_count);
    checksum = SparkMixU64(checksum, request->vocab_size);
    checksum = SparkMixU64(checksum, request->allowed_token_count);
    checksum = SparkMixU64(checksum, request->max_allowed_tokens_per_row);
    checksum = SparkMixU64(checksum, report->unique_allowed_token_count);
    checksum = SparkMixU64(checksum, report->masked_token_count);
    checksum = SparkMixU64(checksum, report->output_checksum);
    checksum = SparkMixU64(checksum, report->invalid_token_count);
    checksum = SparkMixU64(checksum, report->duplicate_token_count);
    checksum = SparkMixU64(checksum, report->empty_row_count);
    return checksum;
}

static SparkStatus SparkValidateStructuredMaskArrays(const SparkCudaStructuredMaskRequest *request, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, const float *input_host, uint64_t input_host_values, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report)
{
    uint64_t logit_count;
    SparkStatus status;

    status = SparkValidateCudaStructuredMaskRequest(request, allowed_offsets, allowed_offset_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaStructuredMaskLogitCount(request);
    if (allowed_tokens == 0 || input_host == 0 || masked_host == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (allowed_token_count != request->allowed_token_count || input_host_values < logit_count || masked_host_values < logit_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

uint64_t SparkComputeCudaStructuredMaskHostChecksum(const float *values, uint64_t value_count)
{
    const uint8_t *bytes;
    uint64_t byte_count;
    uint64_t byte_index;
    uint64_t checksum;

    if (values == 0 || value_count == 0)
    {
        return 0;
    }

    bytes = (const uint8_t *)values;
    byte_count = value_count * (uint64_t)sizeof(float);
    checksum = 0x53504B4D43484B38ull;
    checksum = SparkMixU64(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkMixU64(checksum, bytes[byte_index]);
    }

    return checksum;
}

SparkStatus SparkValidateCudaStructuredMaskRequest(const SparkCudaStructuredMaskRequest *request, const uint32_t *allowed_offsets, uint32_t allowed_offset_count)
{
    uint32_t row_index;
    uint32_t row_start;
    uint32_t row_end;

    if (request == 0 || allowed_offsets == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0 || request->row_count > SPARKPIPE_MAX_PHYSICAL_SLOTS || request->vocab_size == 0 || request->allowed_token_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->max_allowed_tokens_per_row == 0 || request->max_allowed_tokens_per_row > SPARKPIPE_MAX_CONSTRAINT_TOKENS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->mask_value >= 0.0f || request->mask_value != request->mask_value)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_STRUCTURED_MASK_SENTINEL || allowed_offset_count < request->row_count + 1u || allowed_offsets[0] != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (row_index = 0; row_index < request->row_count; ++row_index)
    {
        row_start = allowed_offsets[row_index];
        row_end = allowed_offsets[row_index + 1u];
        if (row_end < row_start || row_end - row_start > request->max_allowed_tokens_per_row || row_end > request->allowed_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if (allowed_offsets[request->row_count] != request->allowed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkApplyStructuredMaskHostReference(const SparkCudaStructuredMaskRequest *request, const float *input_host, uint64_t input_host_values, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report)
{
    uint64_t logit_count;
    uint64_t logit_index;
    uint32_t row_index;
    uint32_t token_index;
    uint32_t token_id;
    uint32_t row_start;
    uint32_t row_end;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateStructuredMaskArrays(request, allowed_offsets, allowed_offset_count, allowed_tokens, allowed_token_count, input_host, input_host_values, masked_host, masked_host_values, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaStructuredMaskLogitCount(request);
    report->logit_count = logit_count;
    report->allowed_token_count = request->allowed_token_count;
    report->host_reference_count = 1u;
    for (logit_index = 0; logit_index < logit_count; ++logit_index)
    {
        masked_host[logit_index] = request->mask_value;
    }
    for (row_index = 0; row_index < request->row_count; ++row_index)
    {
        row_start = allowed_offsets[row_index];
        row_end = allowed_offsets[row_index + 1u];
        if (row_start == row_end)
        {
            report->empty_row_count += 1u;
        }
        for (token_index = row_start; token_index < row_end; ++token_index)
        {
            token_id = allowed_tokens[token_index];
            if (token_id >= request->vocab_size)
            {
                report->invalid_token_count += 1u;
                continue;
            }
            if (SparkStructuredMaskTokenIsDuplicate(allowed_tokens, row_start, token_index, token_id))
            {
                report->duplicate_token_count += 1u;
                continue;
            }
            masked_host[((uint64_t)row_index * (uint64_t)request->vocab_size) + token_id] = input_host[((uint64_t)row_index * (uint64_t)request->vocab_size) + token_id];
            report->unique_allowed_token_count += 1u;
        }
    }
    report->masked_token_count = logit_count - report->unique_allowed_token_count;
    if (request->compute_checksum != 0u)
    {
        report->output_checksum = SparkComputeCudaStructuredMaskHostChecksum(masked_host, logit_count);
        if (request->expected_output_checksum != 0u && request->expected_output_checksum != report->output_checksum)
        {
            report->checksum_mismatch_count += 1u;
        }
    }
    report->trace_checksum = SparkStructuredMaskTraceChecksum(request, report);
    if (report->invalid_token_count != 0u || report->duplicate_token_count != 0u || report->empty_row_count != 0u || report->checksum_mismatch_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaStructuredMaskCandidateDeviceKernels(const SparkCudaStructuredMaskRequest *request, const float *device_input_logits, uint64_t input_value_count, const uint32_t *device_allowed_offsets, uint32_t allowed_offset_count, const uint32_t *device_allowed_tokens, uint32_t allowed_token_count, float *device_candidate_logits, uint32_t candidate_value_count, SparkCudaStructuredMaskReport *device_report)
{
    SparkStatus status;
    uint64_t logit_count;

    status = SparkValidateCudaStructuredMaskRequest(request, device_allowed_offsets, allowed_offset_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    logit_count = SparkCudaStructuredMaskLogitCount(request);
    if (device_input_logits == 0 || device_allowed_tokens == 0 || device_candidate_logits == 0 || device_report == 0 || input_value_count < logit_count || allowed_token_count != request->allowed_token_count || candidate_value_count < request->allowed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->compute_checksum != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaStructuredMaskKernels(const SparkCudaStructuredMaskRequest *request, const float *input_host, uint64_t input_host_values, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateStructuredMaskArrays(request, allowed_offsets, allowed_offset_count, allowed_tokens, allowed_token_count, input_host, input_host_values, masked_host, masked_host_values, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
