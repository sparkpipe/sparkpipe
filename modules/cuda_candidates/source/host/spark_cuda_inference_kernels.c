#include <float.h>
#include <math.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_inference_kernels.h"

static uint64_t SparkCudaInferenceLogitCount(const SparkCudaInferenceUtilityRequest *request)
{
    if (request == 0)
    {
        return 0;
    }

    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

static uint64_t SparkCudaInferenceTraceChecksum(const SparkCudaInferenceUtilityRequest *request, const SparkCudaInferenceUtilityReport *report)
{
    uint64_t checksum;

    checksum = 0x5350494E46545243ull;
    checksum = SparkMixU64(checksum, request->row_count);
    checksum = SparkMixU64(checksum, request->vocab_size);
    checksum = SparkMixU64(checksum, request->scale_count);
    checksum = SparkMixU64(checksum, request->workspace_value_count);
    checksum = SparkMixU64(checksum, report->logits_checksum);
    checksum = SparkMixU64(checksum, report->workspace_checksum);
    checksum = SparkMixU64(checksum, report->token_checksum);
    checksum = SparkMixU64(checksum, report->invalid_scale_count);
    checksum = SparkMixU64(checksum, report->nonfinite_logit_count);
    return checksum;
}

static bool SparkCudaInferenceFloatIsFinite(float value)
{
    return isfinite(value) != 0;
}

static SparkStatus SparkValidateCudaInferenceUtilityArrays(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report)
{
    uint64_t logit_count;
    SparkStatus status;

    status = SparkValidateCudaInferenceUtilityRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaInferenceLogitCount(request);
    if (input_logits == 0 || scale_values == 0 || output_logits == 0 || output_token_ids == 0 || workspace_values == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (input_logit_count < logit_count || output_logit_count < logit_count || output_token_count < request->row_count || scale_count != request->scale_count || workspace_value_count < request->workspace_value_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

uint64_t SparkComputeCudaInferenceFloatHostChecksum(const float *values, uint64_t value_count)
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
    checksum = 0x5350494E46434846ull;
    checksum = SparkMixU64(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkMixU64(checksum, bytes[byte_index]);
    }

    return checksum;
}

uint64_t SparkComputeCudaInferenceU32HostChecksum(const uint32_t *values, uint64_t value_count)
{
    uint64_t value_index;
    uint64_t checksum;

    if (values == 0 || value_count == 0)
    {
        return 0;
    }

    checksum = 0x5350494E46434855ull;
    checksum = SparkMixU64(checksum, value_count);
    for (value_index = 0; value_index < value_count; ++value_index)
    {
        checksum = SparkMixU64(checksum, values[value_index]);
    }

    return checksum;
}

SparkStatus SparkValidateCudaInferenceUtilityRequest(const SparkCudaInferenceUtilityRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0 || request->row_count > SPARKPIPE_MAX_PHYSICAL_SLOTS || request->vocab_size == 0 || request->scale_count == 0 || request->workspace_value_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkCudaInferenceFloatIsFinite(request->temperature) || request->temperature <= 0.0f || !SparkCudaInferenceFloatIsFinite(request->logit_bias) || !SparkCudaInferenceFloatIsFinite(request->workspace_value))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkCudaInferenceFloatIsFinite(request->scale_min) || !SparkCudaInferenceFloatIsFinite(request->scale_max) || request->scale_min <= 0.0f || request->scale_max < request->scale_min)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_INFERENCE_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaInferenceLogitCount(request) == 0 || SparkCudaInferenceLogitCount(request) > (uint64_t)SPARKPIPE_MAX_READY_REQUESTS * (uint64_t)request->vocab_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaInferenceUtilityHostReference(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report)
{
    uint64_t logit_count;
    uint64_t logit_index;
    uint64_t workspace_index;
    uint32_t scale_index;
    uint32_t row_index;
    uint32_t token_index;
    uint32_t best_token_id;
    float scale_value;
    float transformed_value;
    float best_value;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaInferenceUtilityArrays(request, input_logits, input_logit_count, scale_values, scale_count, output_logits, output_logit_count, output_token_ids, output_token_count, workspace_values, workspace_value_count, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    logit_count = SparkCudaInferenceLogitCount(request);
    report->logit_count = logit_count;
    report->scale_count = request->scale_count;
    report->workspace_value_count = request->workspace_value_count;
    report->host_reference_count = 1u;
    for (scale_index = 0; scale_index < request->scale_count; ++scale_index)
    {
        scale_value = scale_values[scale_index];
        if (!SparkCudaInferenceFloatIsFinite(scale_value))
        {
            report->nonfinite_scale_count += 1u;
            report->invalid_scale_count += 1u;
        }
        else if (scale_value == 0.0f)
        {
            report->zero_scale_count += 1u;
            report->invalid_scale_count += 1u;
        }
        else
        {
            if (scale_value < request->scale_min)
            {
                report->scale_underflow_count += 1u;
                report->invalid_scale_count += 1u;
            }
            if (scale_value > request->scale_max)
            {
                report->scale_overflow_count += 1u;
                report->invalid_scale_count += 1u;
            }
        }
    }
    for (logit_index = 0; logit_index < logit_count; ++logit_index)
    {
        transformed_value = (input_logits[logit_index] / request->temperature) + request->logit_bias;
        output_logits[logit_index] = transformed_value;
        if (!SparkCudaInferenceFloatIsFinite(input_logits[logit_index]) || !SparkCudaInferenceFloatIsFinite(transformed_value))
        {
            report->nonfinite_logit_count += 1u;
        }
    }
    for (row_index = 0; row_index < request->row_count; ++row_index)
    {
        best_token_id = 0u;
        best_value = output_logits[(uint64_t)row_index * (uint64_t)request->vocab_size];
        for (token_index = 1u; token_index < request->vocab_size; ++token_index)
        {
            transformed_value = output_logits[((uint64_t)row_index * (uint64_t)request->vocab_size) + token_index];
            if (transformed_value > best_value)
            {
                best_value = transformed_value;
                best_token_id = token_index;
            }
        }
        output_token_ids[row_index] = best_token_id;
    }
    for (workspace_index = 0; workspace_index < request->workspace_value_count; ++workspace_index)
    {
        workspace_values[workspace_index] = request->workspace_value;
    }
    if (request->compute_checksum != 0u)
    {
        report->logits_checksum = SparkComputeCudaInferenceFloatHostChecksum(output_logits, logit_count);
        report->workspace_checksum = SparkComputeCudaInferenceFloatHostChecksum(workspace_values, request->workspace_value_count);
        report->token_checksum = SparkComputeCudaInferenceU32HostChecksum(output_token_ids, request->row_count);
        if ((request->expected_logits_checksum != 0u && request->expected_logits_checksum != report->logits_checksum) || (request->expected_workspace_checksum != 0u && request->expected_workspace_checksum != report->workspace_checksum))
        {
            report->checksum_mismatch_count += 1u;
        }
    }
    report->trace_checksum = SparkCudaInferenceTraceChecksum(request, report);
    if (report->invalid_scale_count != 0u || report->nonfinite_logit_count != 0u || report->checksum_mismatch_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaInferenceUtilityDeviceKernels(const SparkCudaInferenceUtilityRequest *request, const float *device_input_logits, uint64_t input_logit_count, const float *device_scale_values, uint32_t scale_count, float *device_output_logits, uint64_t output_logit_count, uint32_t *device_output_token_ids, uint32_t output_token_count, float *device_workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *device_report)
{
    SparkStatus status;

    status = SparkValidateCudaInferenceUtilityArrays(request, device_input_logits, input_logit_count, device_scale_values, scale_count, device_output_logits, output_logit_count, device_output_token_ids, output_token_count, device_workspace_values, workspace_value_count, device_report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaInferenceUtilityKernels(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaInferenceUtilityArrays(request, input_logits, input_logit_count, scale_values, scale_count, output_logits, output_logit_count, output_token_ids, output_token_count, workspace_values, workspace_value_count, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
