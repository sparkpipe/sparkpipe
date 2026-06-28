#include <math.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_remaining_kernels.h"

static int32_t SparkDecodeFp4Nibble(uint8_t packed_value, uint64_t value_index)
{
    uint8_t nibble;

    nibble = (value_index & 1u) == 0u ? (packed_value & 0x0fu) : ((packed_value >> 4u) & 0x0fu);
    return nibble < 8u ? (int32_t)nibble : (int32_t)nibble - 16;
}

static float SparkReadFp4Value(const uint8_t *packed_values, uint64_t value_index, float scale)
{
    return (float)SparkDecodeFp4Nibble(packed_values[value_index >> 1u], value_index) * scale;
}

static uint64_t SparkRemainingFloatChecksum(const float *values, uint64_t value_count)
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
    checksum = 0x535052454D434846ull;
    checksum = SparkMixU64(checksum, value_count);
    for (byte_index = 0; byte_index < byte_count; ++byte_index)
    {
        checksum = SparkMixU64(checksum, bytes[byte_index]);
    }

    return checksum;
}

static uint64_t SparkRemainingU32Checksum(const uint32_t *values, uint64_t value_count)
{
    uint64_t value_index;
    uint64_t checksum;

    if (values == 0 || value_count == 0)
    {
        return 0;
    }

    checksum = 0x535052454D434855ull;
    checksum = SparkMixU64(checksum, value_count);
    for (value_index = 0; value_index < value_count; ++value_index)
    {
        checksum = SparkMixU64(checksum, values[value_index]);
    }

    return checksum;
}

static uint64_t SparkRemainingTraceChecksum(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceReport *report)
{
    uint64_t checksum;

    checksum = 0x535052454D545243ull;
    checksum = SparkMixU64(checksum, request->token_count);
    checksum = SparkMixU64(checksum, request->hidden_size);
    checksum = SparkMixU64(checksum, request->output_size);
    checksum = SparkMixU64(checksum, request->expert_count);
    checksum = SparkMixU64(checksum, request->query_count);
    checksum = SparkMixU64(checksum, request->kv_count);
    checksum = SparkMixU64(checksum, request->head_dim);
    checksum = SparkMixU64(checksum, report->fp4_linear_checksum);
    checksum = SparkMixU64(checksum, report->sparse_mla_checksum);
    checksum = SparkMixU64(checksum, report->draft_checksum);
    checksum = SparkMixU64(checksum, report->accepted_token_count);
    checksum = SparkMixU64(checksum, report->rejected_token_count);
    return checksum;
}

static SparkStatus SparkValidateRemainingPointers(const SparkCudaRemainingInferenceInputs *inputs, const SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report)
{
    if (inputs == 0 || outputs == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (inputs->token_input == 0 || inputs->fp4_linear_weight == 0 || inputs->expert_weight == 0 || inputs->expert_fp4_weight == 0 || inputs->shared_weight == 0 || inputs->expert_ids == 0 || inputs->query == 0 || inputs->key == 0 || inputs->value == 0 || inputs->sparse_indices == 0 || inputs->physical_block_map == 0 || inputs->draft_hidden == 0 || inputs->draft_weight == 0 || inputs->expected_tokens == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (outputs->fp4_linear_output == 0 || outputs->expert_output == 0 || outputs->expert_fp4_output == 0 || outputs->shared_output == 0 || outputs->mapped_sparse_indices == 0 || outputs->dense_mla_output == 0 || outputs->sparse_mla_output == 0 || outputs->draft_logits == 0 || outputs->draft_token_ids == 0 || outputs->accepted_mask == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkValidateCudaRemainingInferenceRequest(const SparkCudaRemainingInferenceRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0 || request->token_count > SPARKPIPE_MAX_PHYSICAL_SLOTS || request->hidden_size == 0 || request->output_size == 0 || request->expert_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->query_count == 0 || request->kv_count == 0 || request->head_dim == 0 || request->sparse_top_k == 0 || request->sparse_top_k > request->kv_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->draft_count == 0 || request->vocab_size == 0 || !isfinite(request->fp4_scale) || !isfinite(request->expert_fp4_scale) || request->fp4_scale <= 0.0f || request->expert_fp4_scale <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_REMAINING_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaRemainingInferenceHostReference(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *inputs, SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report)
{
    float sum;
    float score;
    float best_score;
    uint32_t token_index;
    uint32_t hidden_index;
    uint32_t output_index;
    uint32_t expert_index;
    uint32_t query_index;
    uint32_t kv_index;
    uint32_t sparse_index;
    uint32_t best_index;
    uint32_t draft_index;
    uint32_t vocab_index;
    uint32_t accepted_prefix;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRemainingInferenceRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateRemainingPointers(inputs, outputs, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    report->host_reference_count = 1u;
    for (token_index = 0; token_index < request->token_count; ++token_index)
    {
        expert_index = inputs->expert_ids[token_index];
        if (expert_index >= request->expert_count)
        {
            report->invalid_expert_count += 1u;
            expert_index = 0u;
        }
        for (output_index = 0; output_index < request->output_size; ++output_index)
        {
            sum = 0.0f;
            for (hidden_index = 0; hidden_index < request->hidden_size; ++hidden_index)
            {
                sum += inputs->token_input[(uint64_t)token_index * request->hidden_size + hidden_index] * SparkReadFp4Value(inputs->fp4_linear_weight, ((uint64_t)hidden_index * request->output_size) + output_index, request->fp4_scale);
            }
            outputs->fp4_linear_output[(uint64_t)token_index * request->output_size + output_index] = sum;
            sum = 0.0f;
            for (hidden_index = 0; hidden_index < request->hidden_size; ++hidden_index)
            {
                sum += inputs->token_input[(uint64_t)token_index * request->hidden_size + hidden_index] * inputs->expert_weight[((uint64_t)expert_index * request->hidden_size * request->output_size) + ((uint64_t)hidden_index * request->output_size) + output_index];
            }
            outputs->expert_output[(uint64_t)token_index * request->output_size + output_index] = sum;
            sum = 0.0f;
            for (hidden_index = 0; hidden_index < request->hidden_size; ++hidden_index)
            {
                sum += inputs->token_input[(uint64_t)token_index * request->hidden_size + hidden_index] * SparkReadFp4Value(inputs->expert_fp4_weight, ((uint64_t)expert_index * request->hidden_size * request->output_size) + ((uint64_t)hidden_index * request->output_size) + output_index, request->expert_fp4_scale);
            }
            outputs->expert_fp4_output[(uint64_t)token_index * request->output_size + output_index] = sum;
            sum = outputs->expert_output[(uint64_t)token_index * request->output_size + output_index];
            for (hidden_index = 0; hidden_index < request->hidden_size; ++hidden_index)
            {
                sum += inputs->token_input[(uint64_t)token_index * request->hidden_size + hidden_index] * inputs->shared_weight[(uint64_t)hidden_index * request->output_size + output_index];
            }
            outputs->shared_output[(uint64_t)token_index * request->output_size + output_index] = sum;
        }
    }
    for (query_index = 0; query_index < request->query_count; ++query_index)
    {
        best_index = 0u;
        best_score = -340282346638528859811704183484516925440.0f;
        for (kv_index = 0; kv_index < request->kv_count; ++kv_index)
        {
            score = 0.0f;
            for (hidden_index = 0; hidden_index < request->head_dim; ++hidden_index)
            {
                score += inputs->query[(uint64_t)query_index * request->head_dim + hidden_index] * inputs->key[(uint64_t)kv_index * request->head_dim + hidden_index];
            }
            if (score > best_score)
            {
                best_score = score;
                best_index = kv_index;
            }
        }
        for (hidden_index = 0; hidden_index < request->head_dim; ++hidden_index)
        {
            outputs->dense_mla_output[(uint64_t)query_index * request->head_dim + hidden_index] = inputs->value[(uint64_t)best_index * request->head_dim + hidden_index];
        }
        best_index = 0u;
        best_score = -340282346638528859811704183484516925440.0f;
        for (sparse_index = 0; sparse_index < request->sparse_top_k; ++sparse_index)
        {
            kv_index = inputs->sparse_indices[(uint64_t)query_index * request->sparse_top_k + sparse_index];
            if (kv_index >= request->kv_count || inputs->physical_block_map[kv_index] >= request->kv_count)
            {
                report->invalid_sparse_index_count += 1u;
                outputs->mapped_sparse_indices[(uint64_t)query_index * request->sparse_top_k + sparse_index] = UINT32_MAX;
                continue;
            }
            outputs->mapped_sparse_indices[(uint64_t)query_index * request->sparse_top_k + sparse_index] = inputs->physical_block_map[kv_index];
            score = 0.0f;
            for (hidden_index = 0; hidden_index < request->head_dim; ++hidden_index)
            {
                score += inputs->query[(uint64_t)query_index * request->head_dim + hidden_index] * inputs->key[(uint64_t)outputs->mapped_sparse_indices[(uint64_t)query_index * request->sparse_top_k + sparse_index] * request->head_dim + hidden_index];
            }
            if (score > best_score)
            {
                best_score = score;
                best_index = outputs->mapped_sparse_indices[(uint64_t)query_index * request->sparse_top_k + sparse_index];
            }
        }
        for (hidden_index = 0; hidden_index < request->head_dim; ++hidden_index)
        {
            outputs->sparse_mla_output[(uint64_t)query_index * request->head_dim + hidden_index] = inputs->value[(uint64_t)best_index * request->head_dim + hidden_index];
        }
    }
    for (draft_index = 0; draft_index < request->draft_count; ++draft_index)
    {
        best_index = 0u;
        best_score = -340282346638528859811704183484516925440.0f;
        for (vocab_index = 0; vocab_index < request->vocab_size; ++vocab_index)
        {
            sum = 0.0f;
            for (hidden_index = 0; hidden_index < request->hidden_size; ++hidden_index)
            {
                sum += inputs->draft_hidden[(uint64_t)draft_index * request->hidden_size + hidden_index] * inputs->draft_weight[(uint64_t)hidden_index * request->vocab_size + vocab_index];
            }
            outputs->draft_logits[(uint64_t)draft_index * request->vocab_size + vocab_index] = sum;
            if (sum > best_score)
            {
                best_score = sum;
                best_index = vocab_index;
            }
        }
        outputs->draft_token_ids[draft_index] = best_index;
    }
    accepted_prefix = 1u;
    for (draft_index = 0; draft_index < request->draft_count; ++draft_index)
    {
        if (accepted_prefix != 0u && outputs->draft_token_ids[draft_index] == inputs->expected_tokens[draft_index])
        {
            outputs->accepted_mask[draft_index] = 1u;
            report->accepted_token_count += 1u;
        }
        else
        {
            outputs->accepted_mask[draft_index] = 0u;
            report->rejected_token_count += 1u;
            accepted_prefix = 0u;
        }
    }
    if (request->compute_checksum != 0u)
    {
        report->fp4_linear_checksum = SparkRemainingFloatChecksum(outputs->fp4_linear_output, (uint64_t)request->token_count * request->output_size);
        report->expert_checksum = SparkRemainingFloatChecksum(outputs->expert_output, (uint64_t)request->token_count * request->output_size);
        report->expert_fp4_checksum = SparkRemainingFloatChecksum(outputs->expert_fp4_output, (uint64_t)request->token_count * request->output_size);
        report->shared_checksum = SparkRemainingFloatChecksum(outputs->shared_output, (uint64_t)request->token_count * request->output_size);
        report->map_checksum = SparkRemainingU32Checksum(outputs->mapped_sparse_indices, (uint64_t)request->query_count * request->sparse_top_k);
        report->dense_mla_checksum = SparkRemainingFloatChecksum(outputs->dense_mla_output, (uint64_t)request->query_count * request->head_dim);
        report->sparse_mla_checksum = SparkRemainingFloatChecksum(outputs->sparse_mla_output, (uint64_t)request->query_count * request->head_dim);
        report->draft_checksum = SparkRemainingFloatChecksum(outputs->draft_logits, (uint64_t)request->draft_count * request->vocab_size);
        report->token_checksum = SparkRemainingU32Checksum(outputs->draft_token_ids, request->draft_count);
        report->accept_checksum = SparkRemainingU32Checksum(outputs->accepted_mask, request->draft_count);
    }
    report->trace_checksum = SparkRemainingTraceChecksum(request, report);
    if (report->invalid_expert_count != 0u || report->invalid_sparse_index_count != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaRemainingInferenceDeviceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *device_inputs, SparkCudaRemainingInferenceOutputs *device_outputs, SparkCudaRemainingInferenceReport *device_report)
{
    SparkStatus status;

    if (device_report != 0)
    {
        memset(device_report, 0, sizeof(*device_report));
    }
    status = SparkValidateCudaRemainingInferenceRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateRemainingPointers(device_inputs, device_outputs, device_report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaRemainingInferenceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *inputs, SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRemainingInferenceRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkValidateRemainingPointers(inputs, outputs, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
