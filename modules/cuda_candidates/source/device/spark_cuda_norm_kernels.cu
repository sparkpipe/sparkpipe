#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_norm_kernels.h"

#define SPARK_CUDA_NORM_THREADS 256u
#define SPARK_CUDA_NORM_HIDDEN_4096 4096u
#define SPARK_CUDA_NORM_VALUES_PER_THREAD_4096 16u
#define SPARK_CUDA_NORM_FP8_E4M3_MAX 448.0f
#define SPARK_CUDA_NORM_FP8_MIN_SCALE (1.0f / (448.0f * 512.0f))

static __device__ float SparkCudaNormBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaNormFloatToBf16(float value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static __device__ float SparkCudaNormWarpSum(float value)
{
    uint32_t offset;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

static __device__ float SparkCudaNormWarpMax(float value)
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

static __device__ float SparkCudaNormBlockSum(float value, float *warp_values)
{
    uint32_t lane;
    uint32_t warp;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    value = SparkCudaNormWarpSum(value);
    if (lane == 0u)
    {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8u ? warp_values[threadIdx.x] : 0.0f;
    if (warp == 0u)
    {
        value = SparkCudaNormWarpSum(value);
    }
    if (threadIdx.x == 0u)
    {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

static __device__ float SparkCudaNormBlockMax(float value, float *warp_values)
{
    uint32_t lane;
    uint32_t warp;

    lane = threadIdx.x & 31u;
    warp = threadIdx.x >> 5u;
    value = SparkCudaNormWarpMax(value);
    if (lane == 0u)
    {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8u ? warp_values[threadIdx.x] : 0.0f;
    if (warp == 0u)
    {
        value = SparkCudaNormWarpMax(value);
    }
    if (threadIdx.x == 0u)
    {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

static __global__ void SparkCudaRmsNormBf16Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *weight_values, uint16_t *output_values)
{
    __shared__ float partial_sums[8u];
    uint32_t row_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float value;
    float sum;
    float scale;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }
    row_offset = (uint64_t)row_index * (uint64_t)request.hidden_size;
    sum = 0.0f;
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        sum += value * value;
    }
    scale = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) / (float)request.hidden_size) + request.epsilon);
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        value = value * scale * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
    }
}

static __global__ void SparkCudaFusedAddRmsNormBf16Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *residual_values, const uint16_t *weight_values, uint16_t *output_values, uint16_t *residual_output_values)
{
    __shared__ float partial_sums[8u];
    uint32_t row_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float value;
    float sum;
    float scale;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }
    row_offset = (uint64_t)row_index * (uint64_t)request.hidden_size;
    sum = 0.0f;
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]) + SparkCudaNormBf16ToFloat(residual_values[row_offset + hidden_index]);
        sum += value * value;
        residual_output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
    }
    scale = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) / (float)request.hidden_size) + request.epsilon);
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(residual_output_values[row_offset + hidden_index]);
        value = value * scale * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
    }
}

static __global__ void SparkCudaRmsNormQuantFp8E4m3Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *weight_values, uint8_t *output_values, float *output_scales)
{
    __shared__ float partial_sums[8u];
    __shared__ float partial_maxes[8u];
    uint32_t row_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float value;
    float sum;
    float inverse_rms;
    float abs_value;
    float row_max;
    float scale;

    row_index = blockIdx.x;
    if (row_index >= request.row_count)
    {
        return;
    }
    row_offset = (uint64_t)row_index * (uint64_t)request.hidden_size;
    sum = 0.0f;
    abs_value = 0.0f;
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        sum += value * value;
        abs_value = fmaxf(abs_value, fabsf(value * SparkCudaNormBf16ToFloat(weight_values[hidden_index])));
    }
    inverse_rms = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) / (float)request.hidden_size) + request.epsilon);
    row_max = inverse_rms * SparkCudaNormBlockMax(abs_value, partial_maxes);
    scale = row_max / SPARK_CUDA_NORM_FP8_E4M3_MAX;
    if (scale < SPARK_CUDA_NORM_FP8_MIN_SCALE)
    {
        scale = SPARK_CUDA_NORM_FP8_MIN_SCALE;
    }
    if (threadIdx.x == 0u)
    {
        output_scales[row_index] = scale;
    }
    for (hidden_index = threadIdx.x; hidden_index < request.hidden_size; hidden_index += blockDim.x)
    {
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        value = value * inverse_rms * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        output_values[row_offset + hidden_index] = (uint8_t)__nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3);
    }
}

static __global__ void SparkCudaRmsNormBf16Hidden4096Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *weight_values, uint16_t *output_values)
{
    __shared__ float partial_sums[8u];
    uint32_t row_index;
    uint32_t value_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float cached_values[SPARK_CUDA_NORM_VALUES_PER_THREAD_4096];
    float value;
    float sum;
    float scale;

    row_index = blockIdx.x;
    row_offset = (uint64_t)row_index * (uint64_t)SPARK_CUDA_NORM_HIDDEN_4096;
    sum = 0.0f;
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        cached_values[value_index] = value;
        sum += value * value;
    }
    scale = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) * (1.0f / 4096.0f)) + request.epsilon);
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        value = cached_values[value_index] * scale * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
    }
}

static __global__ void SparkCudaFusedAddRmsNormBf16Hidden4096Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *residual_values, const uint16_t *weight_values, uint16_t *output_values, uint16_t *residual_output_values)
{
    __shared__ float partial_sums[8u];
    uint32_t row_index;
    uint32_t value_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float cached_values[SPARK_CUDA_NORM_VALUES_PER_THREAD_4096];
    float value;
    float sum;
    float scale;

    row_index = blockIdx.x;
    row_offset = (uint64_t)row_index * (uint64_t)SPARK_CUDA_NORM_HIDDEN_4096;
    sum = 0.0f;
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]) + SparkCudaNormBf16ToFloat(residual_values[row_offset + hidden_index]);
        cached_values[value_index] = value;
        sum += value * value;
    }
    scale = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) * (1.0f / 4096.0f)) + request.epsilon);
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        value = cached_values[value_index];
        residual_output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
        value = value * scale * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        output_values[row_offset + hidden_index] = SparkCudaNormFloatToBf16(value);
    }
}

static __global__ void SparkCudaRmsNormQuantFp8E4m3Hidden4096Kernel(SparkCudaNormRequest request, const uint16_t *input_values, const uint16_t *weight_values, uint8_t *output_values, float *output_scales)
{
    __shared__ float partial_sums[8u];
    __shared__ float partial_maxes[8u];
    uint32_t row_index;
    uint32_t value_index;
    uint32_t hidden_index;
    uint64_t row_offset;
    float cached_values[SPARK_CUDA_NORM_VALUES_PER_THREAD_4096];
    float value;
    float sum;
    float max_value;
    float inverse_rms;
    float scale;

    row_index = blockIdx.x;
    row_offset = (uint64_t)row_index * (uint64_t)SPARK_CUDA_NORM_HIDDEN_4096;
    sum = 0.0f;
    max_value = 0.0f;
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        value = SparkCudaNormBf16ToFloat(input_values[row_offset + hidden_index]);
        sum += value * value;
        value = value * SparkCudaNormBf16ToFloat(weight_values[hidden_index]);
        cached_values[value_index] = value;
        max_value = fmaxf(max_value, fabsf(value));
    }
    inverse_rms = rsqrtf((SparkCudaNormBlockSum(sum, partial_sums) * (1.0f / 4096.0f)) + request.epsilon);
    scale = (inverse_rms * SparkCudaNormBlockMax(max_value, partial_maxes)) / SPARK_CUDA_NORM_FP8_E4M3_MAX;
    if (scale < SPARK_CUDA_NORM_FP8_MIN_SCALE)
    {
        scale = SPARK_CUDA_NORM_FP8_MIN_SCALE;
    }
    if (threadIdx.x == 0u)
    {
        output_scales[row_index] = scale;
    }
#pragma unroll
    for (value_index = 0u; value_index < SPARK_CUDA_NORM_VALUES_PER_THREAD_4096; ++value_index)
    {
        hidden_index = (value_index * SPARK_CUDA_NORM_THREADS) + threadIdx.x;
        output_values[row_offset + hidden_index] = (uint8_t)__nv_cvt_float_to_fp8((cached_values[value_index] * inverse_rms) / scale, __NV_SATFINITE, __NV_E4M3);
    }
}

extern "C" SparkStatus SparkRunCudaRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_bf16, SparkCudaNormReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_weight_bf16 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->hidden_size == SPARK_CUDA_NORM_HIDDEN_4096)
    {
        SparkCudaRmsNormBf16Hidden4096Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_weight_bf16, (uint16_t *)device_output_bf16);
    }
    else
    {
        SparkCudaRmsNormBf16Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_weight_bf16, (uint16_t *)device_output_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->rms_norm_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaFusedAddRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_residual_bf16, const void *device_weight_bf16, void *device_output_bf16, void *device_residual_output_bf16, SparkCudaNormReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_residual_bf16 == 0 || device_weight_bf16 == 0 || device_output_bf16 == 0 || device_residual_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->hidden_size == SPARK_CUDA_NORM_HIDDEN_4096)
    {
        SparkCudaFusedAddRmsNormBf16Hidden4096Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_residual_bf16, (const uint16_t *)device_weight_bf16, (uint16_t *)device_output_bf16, (uint16_t *)device_residual_output_bf16);
    }
    else
    {
        SparkCudaFusedAddRmsNormBf16Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_residual_bf16, (const uint16_t *)device_weight_bf16, (uint16_t *)device_output_bf16, (uint16_t *)device_residual_output_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->fused_add_rms_norm_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaRmsNormQuantFp8E4m3(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_fp8, float *device_output_scales, SparkCudaNormReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_weight_bf16 == 0 || device_output_fp8 == 0 || device_output_scales == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->hidden_size == SPARK_CUDA_NORM_HIDDEN_4096)
    {
        SparkCudaRmsNormQuantFp8E4m3Hidden4096Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_weight_bf16, (uint8_t *)device_output_fp8, device_output_scales);
    }
    else
    {
        SparkCudaRmsNormQuantFp8E4m3Kernel<<<request->row_count, SPARK_CUDA_NORM_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (const uint16_t *)device_weight_bf16, (uint8_t *)device_output_fp8, device_output_scales);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->rms_norm_fp8_quant_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
