#include <math.h>
#include <string.h>

#include "sparkpipe/spark_cuda_fp4_gemm.h"
#include "sparkpipe/spark_cuda_fp4_quant.h"

static float SparkCudaFp4QuantBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static void SparkCudaFp4QuantPackNibble(uint8_t *values, uint32_t value_index, uint8_t nibble)
{
    uint32_t byte_index;

    byte_index = value_index >> 1u;
    if ((value_index & 1u) == 0u)
        values[byte_index] = (uint8_t)((values[byte_index] & 0xf0u) | (nibble & 0x0fu));
    else
        values[byte_index] = (uint8_t)((values[byte_index] & 0x0fu) | ((nibble & 0x0fu) << 4u));
}

static void SparkCudaFp4QuantFillReportShape(const SparkCudaFp4QuantRequest *request, SparkCudaFp4QuantReport *report)
{
    if (request == 0 || report == 0)
        return;
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->col_count;
    report->packed_bytes = SparkCudaFp4QuantPackedBytes(request->row_count, request->col_count);
    report->scale_bytes = SparkCudaFp4QuantScaleBytes(request->row_count, request->col_count);
    report->row_count = request->row_count;
    report->col_count = request->col_count;
    report->scale_block_count = request->col_count / SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK;
    report->global_scale = request->global_scale;
}

SparkStatus SparkValidateCudaFp4QuantRequest(const SparkCudaFp4QuantRequest *request)
{
    if (request == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->sentinel != SPARKPIPE_CUDA_FP4_QUANT_SENTINEL)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->row_count == 0u || request->col_count == 0u || (request->col_count % SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK) != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->global_scale <= 0.0f || request->global_scale != request->global_scale || !isfinite(request->global_scale))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkCudaFp4QuantPackedBytes(request->row_count, request->col_count) == 0u || SparkCudaFp4QuantScaleBytes(request->row_count, request->col_count) == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

uint64_t SparkCudaFp4QuantPackedBytes(uint32_t rows, uint32_t cols)
{
    return SparkCudaFp4GemmPackedBytes(rows, cols);
}

uint64_t SparkCudaFp4QuantScaleBytes(uint32_t rows, uint32_t cols)
{
    return SparkCudaFp4GemmScaleBytes(rows, cols);
}

float SparkCudaFp4QuantDefaultGlobalScale(float absolute_max)
{
    if (absolute_max <= 0.0f || absolute_max != absolute_max || !isfinite(absolute_max))
        return 1.0f;
    return (SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX * SPARKPIPE_CUDA_FP4_QUANT_FP4_MAX) / absolute_max;
}

float SparkCudaFp4QuantDecodeE2m1Host(uint8_t nibble)
{
    static const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value;

    value = values[nibble & 7u];
    return (nibble & 8u) != 0u ? -value : value;
}

uint8_t SparkCudaFp4QuantEncodeE2m1Host(float value)
{
    static const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float abs_value;
    float best_delta;
    uint8_t best_index;
    uint8_t value_index;
    uint8_t sign;

    if (value != value || !isfinite(value))
        return 0u;
    sign = value < 0.0f ? 8u : 0u;
    abs_value = fabsf(value);
    best_index = 0u;
    best_delta = fabsf(abs_value - values[0]);
    for (value_index = 1u; value_index < 8u; ++value_index)
    {
        float delta;

        delta = fabsf(abs_value - values[value_index]);
        if (delta < best_delta)
        {
            best_delta = delta;
            best_index = value_index;
        }
    }
    return (uint8_t)(sign | best_index);
}

float SparkCudaFp4QuantDecodeE4m3Host(uint8_t byte_value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    float value;

    sign = byte_value & 0x80u;
    exponent = (byte_value >> 3u) & 0x0fu;
    mantissa = byte_value & 0x07u;
    if (exponent == 0u)
        value = mantissa == 0u ? 0.0f : ldexpf((float)mantissa / 8.0f, -6);
    else if (exponent == 15u && mantissa >= 7u)
        value = SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX;
    else
        value = ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    return sign != 0u ? -value : value;
}

uint8_t SparkCudaFp4QuantEncodeE4m3Host(float value)
{
    float abs_value;
    float scaled;
    uint32_t sign;
    int32_t exponent;
    int32_t exponent_field;
    int32_t mantissa;

    if (value != value)
        return 0u;
    sign = value < 0.0f ? 0x80u : 0u;
    abs_value = fabsf(value);
    if (abs_value == 0.0f)
        return (uint8_t)sign;
    if (!isfinite(abs_value) || abs_value >= SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX)
        return (uint8_t)(sign | 0x7eu);
    if (abs_value < ldexpf(1.0f, -6))
    {
        mantissa = (int32_t)floorf((abs_value / ldexpf(1.0f, -9)) + 0.5f);
        if (mantissa <= 0)
            return (uint8_t)sign;
        if (mantissa > 7)
            mantissa = 7;
        return (uint8_t)(sign | (uint32_t)mantissa);
    }
    exponent = (int32_t)floorf(log2f(abs_value));
    exponent_field = exponent + 7;
    if (exponent_field <= 0)
        return (uint8_t)sign;
    scaled = abs_value / ldexpf(1.0f, exponent);
    mantissa = (int32_t)floorf(((scaled - 1.0f) * 8.0f) + 0.5f);
    if (mantissa >= 8)
    {
        mantissa = 0;
        exponent_field += 1;
    }
    if (exponent_field >= 15)
    {
        if (mantissa > 6 || exponent_field > 15)
            return (uint8_t)(sign | 0x7eu);
    }
    return (uint8_t)(sign | ((uint32_t)exponent_field << 3u) | (uint32_t)mantissa);
}

SparkStatus SparkRunHostBf16ToFp4E2m1(const SparkCudaFp4QuantRequest *request, const uint16_t *input_bf16, uint8_t *output_fp4, uint8_t *output_scales_ue4m3, SparkCudaFp4QuantReport *report)
{
    uint32_t blocks_per_row;
    uint32_t row_index;
    SparkStatus status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateCudaFp4QuantRequest(request);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_bf16 == 0 || output_fp4 == 0 || output_scales_ue4m3 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    SparkCudaFp4QuantFillReportShape(request, report);
    memset(output_fp4, 0, (size_t)report->packed_bytes);
    memset(output_scales_ue4m3, 0, (size_t)report->scale_bytes);
    blocks_per_row = request->col_count / SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK;
    for (row_index = 0u; row_index < request->row_count; ++row_index)
    {
        uint32_t block_index;

        for (block_index = 0u; block_index < blocks_per_row; ++block_index)
        {
            uint32_t block_offset;
            float block_max;
            float scale_value;
            float decoded_scale;
            uint32_t value_index;

            block_offset = (row_index * request->col_count) + (block_index * SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK);
            block_max = 0.0f;
            for (value_index = 0u; value_index < SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK; ++value_index)
                block_max = fmaxf(block_max, fabsf(SparkCudaFp4QuantBf16ToFloat(input_bf16[block_offset + value_index])));
            scale_value = block_max == 0.0f ? 0.0f : ((block_max * request->global_scale) / SPARKPIPE_CUDA_FP4_QUANT_FP4_MAX);
            if (scale_value > SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX)
            {
                scale_value = SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX;
                report->saturated_scale_count += 1u;
            }
            output_scales_ue4m3[(row_index * blocks_per_row) + block_index] = SparkCudaFp4QuantEncodeE4m3Host(scale_value);
            decoded_scale = SparkCudaFp4QuantDecodeE4m3Host(output_scales_ue4m3[(row_index * blocks_per_row) + block_index]);
            for (value_index = 0u; value_index < SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK; ++value_index)
            {
                float value;
                uint8_t nibble;

                value = SparkCudaFp4QuantBf16ToFloat(input_bf16[block_offset + value_index]);
                nibble = decoded_scale == 0.0f ? 0u : SparkCudaFp4QuantEncodeE2m1Host((value * request->global_scale) / decoded_scale);
                SparkCudaFp4QuantPackNibble(output_fp4, block_offset + value_index, nibble);
            }
        }
    }
    report->host_reference_count = 1u;
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaBf16ToFp4E2m1(const SparkCudaFp4QuantRequest *request, const void *device_input_bf16, void *device_output_fp4, void *device_output_scales_ue4m3, SparkCudaFp4QuantReport *report)
{
    SparkStatus status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateCudaFp4QuantRequest(request);
    if (status != SPARK_STATUS_OK)
        return status;
    if (device_input_bf16 == 0 || device_output_fp4 == 0 || device_output_scales_ue4m3 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    SparkCudaFp4QuantFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
