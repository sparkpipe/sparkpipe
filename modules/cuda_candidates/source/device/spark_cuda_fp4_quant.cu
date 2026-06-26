#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_fp4_quant.h"

#define SPARK_CUDA_FP4_QUANT_THREADS 32u

static __device__ float SparkCudaFp4QuantDeviceBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ float SparkCudaFp4QuantDeviceWarpMax(float value)
{
    uint32_t offset;
    float other;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}

static __device__ float SparkCudaFp4QuantDeviceDecodeE4m3(uint8_t byte_value)
{
    uint32_t exponent;
    uint32_t mantissa;
    float value;

    exponent = (byte_value >> 3u) & 0x0fu;
    mantissa = byte_value & 0x07u;
    if (exponent == 0u)
        value = mantissa == 0u ? 0.0f : ldexpf((float)mantissa / 8.0f, -6);
    else if (exponent == 15u && mantissa >= 7u)
        value = SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX;
    else
        value = ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    return (byte_value & 0x80u) != 0u ? -value : value;
}

static __device__ uint8_t SparkCudaFp4QuantDeviceEncodeE2m1(float value)
{
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float abs_value;
    float best_delta;
    uint8_t best_index;
    uint8_t value_index;
    uint8_t sign;

    if (!isfinite(value))
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

static __global__ void SparkCudaBf16ToFp4E2m1Kernel(SparkCudaFp4QuantRequest request, const uint16_t *input_values, uint8_t *output_values, uint8_t *output_scales)
{
    __shared__ float block_scale;
    __shared__ uint8_t block_scale_byte;
    uint32_t blocks_per_row;
    uint32_t block_id;
    uint32_t row_index;
    uint32_t scale_block_index;
    uint32_t block_offset;
    float local_max;
    float max_value;

    blocks_per_row = request.col_count / SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK;
    block_id = blockIdx.x;
    row_index = block_id / blocks_per_row;
    scale_block_index = block_id - (row_index * blocks_per_row);
    if (row_index >= request.row_count)
        return;
    block_offset = (row_index * request.col_count) + (scale_block_index * SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK);
    local_max = 0.0f;
    if (threadIdx.x < SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK)
        local_max = fabsf(SparkCudaFp4QuantDeviceBf16ToFloat(input_values[block_offset + threadIdx.x]));
    max_value = SparkCudaFp4QuantDeviceWarpMax(local_max);
    if (threadIdx.x == 0u)
    {
        float scale_value;

        scale_value = max_value == 0.0f ? 0.0f : ((max_value * request.global_scale) / SPARKPIPE_CUDA_FP4_QUANT_FP4_MAX);
        if (scale_value > SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX)
            scale_value = SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX;
        block_scale_byte = (uint8_t)__nv_cvt_float_to_fp8(scale_value, __NV_SATFINITE, __NV_E4M3);
        block_scale = SparkCudaFp4QuantDeviceDecodeE4m3(block_scale_byte);
        output_scales[(row_index * blocks_per_row) + scale_block_index] = block_scale_byte;
    }
    __syncthreads();
    if (threadIdx.x < 8u)
    {
        uint32_t low_index;
        uint32_t high_index;
        float low_value;
        float high_value;
        uint8_t low_nibble;
        uint8_t high_nibble;

        low_index = block_offset + (threadIdx.x * 2u);
        high_index = low_index + 1u;
        low_value = SparkCudaFp4QuantDeviceBf16ToFloat(input_values[low_index]);
        high_value = SparkCudaFp4QuantDeviceBf16ToFloat(input_values[high_index]);
        low_nibble = block_scale == 0.0f ? 0u : SparkCudaFp4QuantDeviceEncodeE2m1((low_value * request.global_scale) / block_scale);
        high_nibble = block_scale == 0.0f ? 0u : SparkCudaFp4QuantDeviceEncodeE2m1((high_value * request.global_scale) / block_scale);
        output_values[low_index >> 1u] = (uint8_t)(low_nibble | (high_nibble << 4u));
    }
}

extern "C" SparkStatus SparkRunCudaBf16ToFp4E2m1(const SparkCudaFp4QuantRequest *request, const void *device_input_bf16, void *device_output_fp4, void *device_output_scales_ue4m3, SparkCudaFp4QuantReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint32_t blocks_per_row;
    uint32_t grid_blocks;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateCudaFp4QuantRequest(request);
    if (status != SPARK_STATUS_OK)
        return status;
    if (device_input_bf16 == 0 || device_output_fp4 == 0 || device_output_scales_ue4m3 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->col_count;
    report->packed_bytes = SparkCudaFp4QuantPackedBytes(request->row_count, request->col_count);
    report->scale_bytes = SparkCudaFp4QuantScaleBytes(request->row_count, request->col_count);
    report->row_count = request->row_count;
    report->col_count = request->col_count;
    report->scale_block_count = request->col_count / SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK;
    report->global_scale = request->global_scale;
    if (request->sentinel != SPARKPIPE_CUDA_FP4_QUANT_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_status = cudaMemset(device_output_scales_ue4m3, 0, (size_t)report->scale_bytes);
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    blocks_per_row = request->col_count / SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK;
    grid_blocks = request->row_count * blocks_per_row;
    SparkCudaBf16ToFp4E2m1Kernel<<<grid_blocks, SPARK_CUDA_FP4_QUANT_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (uint8_t *)device_output_fp4, (uint8_t *)device_output_scales_ue4m3);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    report->quant_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
