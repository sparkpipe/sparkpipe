#include "spark_glm52_resident_decode_stage_backend.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#ifndef SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT 1
#endif

#ifndef SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_NATIVE_BLOCK_SCALED_MMA
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_NATIVE_BLOCK_SCALED_MMA 0
#endif

#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
#include <cublasLt.h>
#endif

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT UINT32_MAX
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP4_MMA_K 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_M 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_N 8u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_SEQUENCE_GROUP 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_WARPS 4u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_THREADS 128u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_WORKSPACE_ALIGNMENT 256ull
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS 32u

#ifndef SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE
#if defined(__GNUC__) || defined(__clang__)
#define SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE __attribute__((used))
#else
#define SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE
#endif
#endif

typedef SparkStatus (*SparkGlm52Sm121RequiredB12xCreateFunction)(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out);

typedef SparkStatus (*SparkGlm52Sm121RequiredB12xLaunchFunction)(
    void *state,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments);

typedef void (*SparkGlm52Sm121RequiredB12xDestroyFunction)(void *state);

static SparkGlm52Sm121RequiredB12xCreateFunction
    SparkGlm52Sm121RequiredDecodeStageB12xCreateReference
    SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE =
        SparkGlm52Sm121FlashInferB12xMoeCreate;

static SparkGlm52Sm121RequiredB12xLaunchFunction
    SparkGlm52Sm121RequiredDecodeStageB12xLaunchReference
    SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE =
        SparkGlm52Sm121FlashInferB12xMoeLaunch;

static SparkGlm52Sm121RequiredB12xDestroyFunction
    SparkGlm52Sm121RequiredDecodeStageB12xDestroyReference
    SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE =
        SparkGlm52Sm121FlashInferB12xMoeDestroy;


static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageBf16ToFloat(
    uint16_t value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;

    conversion.bits = ((uint32_t)value) << 16u;
    return conversion.value;
}

static __device__ __forceinline__ uint16_t SparkGlm52ResidentDecodeStageFloatToBf16(
    float value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;
    uint32_t rounding_bias;

    conversion.value = value;
    rounding_bias = 0x7fffu + ((conversion.bits >> 16u) & 1u);
    return (uint16_t)((conversion.bits + rounding_bias) >> 16u);
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
    uint8_t value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    float result;

    sign = (uint32_t)(value >> 7u);
    exponent = (uint32_t)((value >> 3u) & 15u);
    mantissa = (uint32_t)(value & 7u);
    if ((value & 0x7fu) == 0u)
    {
        return 0.0f;
    }
    if (exponent == 0u)
    {
        result = ldexpf((float)mantissa / 8.0f, -6);
    }
    else
    {
        result = ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    }
    return sign != 0u ? -result : result;
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageWarpReduceSum(
    float value)
{
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}


static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageWarpReduceMax(
    float value)
{
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 1));
    return value;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageWarpBroadcastU32(
    uint32_t value,
    uint32_t source_lane)
{
    return __shfl_sync(0xffffffffu, value, source_lane);
}

static __device__ float SparkGlm52ResidentDecodeStageBlockReduceSum(
    float value,
    float *shared_reduction)
{
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;

    lane_index = threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index = threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    warp_count = (blockDim.x +
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;

    value = SparkGlm52ResidentDecodeStageWarpReduceSum(value);
    if (lane_index == 0u)
    {
        shared_reduction[warp_index] = value;
    }
    __syncthreads();

    value = threadIdx.x < warp_count ? shared_reduction[lane_index] : 0.0f;
    if (warp_index == 0u)
    {
        value = SparkGlm52ResidentDecodeStageWarpReduceSum(value);
        if (lane_index == 0u)
        {
            shared_reduction[0] = value;
        }
    }
    __syncthreads();
    return shared_reduction[0];
}

static __device__ float SparkGlm52ResidentDecodeStageBlockReduceMax(
    float value,
    float *shared_reduction)
{
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;

    lane_index = threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index = threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    warp_count = (blockDim.x +
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;

    value = SparkGlm52ResidentDecodeStageWarpReduceMax(value);
    if (lane_index == 0u)
    {
        shared_reduction[warp_index] = value;
    }
    __syncthreads();

    value = threadIdx.x < warp_count ? shared_reduction[lane_index] : -FLT_MAX;
    if (warp_index == 0u)
    {
        value = SparkGlm52ResidentDecodeStageWarpReduceMax(value);
        if (lane_index == 0u)
        {
            shared_reduction[0] = value;
        }
    }
    __syncthreads();
    return shared_reduction[0];
}

static __device__ __forceinline__ void SparkGlm52ResidentDecodeStageApplyRopePair(
    uint16_t first_input_bf16,
    uint16_t second_input_bf16,
    float cosine,
    float sine,
    uint16_t *first_output_bf16,
    uint16_t *second_output_bf16)
{
    float first_input;
    float second_input;

    first_input = SparkGlm52ResidentDecodeStageBf16ToFloat(first_input_bf16);
    second_input = SparkGlm52ResidentDecodeStageBf16ToFloat(second_input_bf16);
    *first_output_bf16 = SparkGlm52ResidentDecodeStageFloatToBf16(
        (first_input * cosine) - (second_input * sine));
    *second_output_bf16 = SparkGlm52ResidentDecodeStageFloatToBf16(
        (first_input * sine) + (second_input * cosine));
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageDecodeE2m1(
    uint8_t nibble)
{
    float value;

    switch (nibble & 7u)
    {
        case 1u:
        {
            value = 0.5f;
            break;
        }
        case 2u:
        {
            value = 1.0f;
            break;
        }
        case 3u:
        {
            value = 1.5f;
            break;
        }
        case 4u:
        {
            value = 2.0f;
            break;
        }
        case 5u:
        {
            value = 3.0f;
            break;
        }
        case 6u:
        {
            value = 4.0f;
            break;
        }
        case 7u:
        {
            value = 6.0f;
            break;
        }
        default:
        {
            value = 0.0f;
            break;
        }
    }
    return (nibble & 8u) != 0u ? -value : value;
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageDecodeE8m0(
    uint8_t value)
{
    if (value == 0u)
    {
        return 0.0f;
    }
    return ldexpf(1.0f, (int)value - 127);
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageDecodeUe4m3(
    uint8_t value)
{
    return SparkGlm52ResidentDecodeStageFp8E4m3ToFloat((uint8_t)(value & 0x7fu));
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodeE8m0Saturate(
    float value)
{
    int exponent;
    float mantissa;

    if (!isfinite(value) || value <= 0.0f)
    {
        return 0u;
    }
    mantissa = frexpf(value, &exponent);
    if (mantissa >= 0.7071067811865476f)
    {
        ++exponent;
    }
    exponent += 126;
    if (exponent < 1)
    {
        return 1u;
    }
    if (exponent > 255)
    {
        return 255u;
    }
    return (uint8_t)exponent;
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodeUe4m3Saturate(
    float value)
{
    int exponent;
    float normalized;
    uint32_t biased_exponent;
    uint32_t mantissa;

    if (!isfinite(value) || value <= 0.0f)
    {
        return 0u;
    }
    normalized = frexpf(value, &exponent) * 2.0f;
    --exponent;
    biased_exponent = (uint32_t)(exponent + 7);
    if ((int32_t)biased_exponent <= 0)
    {
        return 1u;
    }
    if (biased_exponent >= 15u)
    {
        return 0x7fu;
    }
    mantissa = (uint32_t)rintf((normalized - 1.0f) * 8.0f);
    if (mantissa >= 8u)
    {
        mantissa = 0u;
        ++biased_exponent;
        if (biased_exponent >= 15u)
        {
            return 0x7fu;
        }
    }
    return (uint8_t)((biased_exponent << 3u) | mantissa);
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(
    float value)
{
    uint32_t sign;
    int exponent;
    float magnitude;
    float normalized;
    uint32_t biased_exponent;
    uint32_t mantissa;

    if (!isfinite(value) || value == 0.0f)
    {
        return 0u;
    }
    sign = value < 0.0f ? 0x80u : 0u;
    magnitude = fabsf(value);
    normalized = frexpf(magnitude, &exponent) * 2.0f;
    --exponent;
    biased_exponent = (uint32_t)(exponent + 7);
    if ((int32_t)biased_exponent <= 0)
    {
        mantissa = (uint32_t)rintf(ldexpf(magnitude, 9));
        if (mantissa > 7u)
        {
            mantissa = 7u;
        }
        return (uint8_t)(sign | mantissa);
    }
    if (biased_exponent >= 15u)
    {
        return (uint8_t)(sign | 0x7eu);
    }
    mantissa = (uint32_t)rintf((normalized - 1.0f) * 8.0f);
    if (mantissa >= 8u)
    {
        mantissa = 0u;
        ++biased_exponent;
        if (biased_exponent >= 15u)
        {
            return (uint8_t)(sign | 0x7eu);
        }
    }
    return (uint8_t)(sign | (biased_exponent << 3u) | mantissa);
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodeE2m1Saturate(
    float value)
{
    static const float code_values[8] = {
        0.0f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        3.0f,
        4.0f,
        6.0f
    };
    float magnitude;
    float best_error;
    uint32_t best_code;
    uint32_t code;

    if (!isfinite(value) || value == 0.0f)
    {
        return 0u;
    }
    magnitude = fabsf(value);
    best_code = 0u;
    best_error = fabsf(magnitude - code_values[0]);
    for (code = 1u; code < 8u; ++code)
    {
        float error;

        error = fabsf(magnitude - code_values[code]);
        if (error < best_error)
        {
            best_error = error;
            best_code = code;
        }
    }
    return (uint8_t)(best_code | (value < 0.0f ? 8u : 0u));
}

static __global__ void SparkGlm52ResidentDecodeStageMarkPhaseKernel(
    uint64_t *phase_clock_cycles,
    uint32_t phase_index)
{
    uint64_t global_time;

    if (phase_clock_cycles != 0 && blockIdx.x == 0u && threadIdx.x == 0u &&
        phase_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT)
    {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
        phase_clock_cycles[phase_index] = global_time;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageClearU32Kernel(
    uint32_t *values,
    uint32_t value_count)
{
    uint32_t value_index;

    value_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (value_index < value_count)
    {
        values[value_index] = 0u;
        value_index += gridDim.x * blockDim.x;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageRmsNormKernel(
    const uint16_t *__restrict__ input_hidden_bf16,
    const uint16_t *__restrict__ weight_bf16,
    uint16_t *__restrict__ output_hidden_bf16,
    uint32_t active_sequence_count,
    float epsilon)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t sequence_index;
    uint32_t hidden_index;
    float local_square_sum;
    float row_square_sum;
    float inverse_rms;

    sequence_index = blockIdx.x;
    if (sequence_index >= active_sequence_count)
    {
        return;
    }
    local_square_sum = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float value;

        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_hidden_bf16[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                (uint64_t)hidden_index]);
        local_square_sum += value * value;
    }
    row_square_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_square_sum,
        shared_reduction);
    inverse_rms = rsqrtf(
        (row_square_sum /
         (float)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
        epsilon);
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float value;
        float weight;

        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_hidden_bf16[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                (uint64_t)hidden_index]);
        weight = SparkGlm52ResidentDecodeStageBf16ToFloat(
            weight_bf16[hidden_index]);
        output_hidden_bf16[
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
            (uint64_t)hidden_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(value * inverse_rms * weight);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageRmsNormDimensionKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t active_sequence_count,
    uint32_t dimension,
    float epsilon)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t sequence_index;
    uint32_t dimension_index;
    float local_square_sum;
    float row_square_sum;
    float inverse_rms;

    sequence_index = blockIdx.x;
    if (sequence_index >= active_sequence_count)
    {
        return;
    }
    local_square_sum = 0.0f;
    for (dimension_index = threadIdx.x;
         dimension_index < dimension;
         dimension_index += blockDim.x)
    {
        float value;

        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)dimension) +
                (uint64_t)dimension_index]);
        local_square_sum += value * value;
    }
    row_square_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_square_sum,
        shared_reduction);
    inverse_rms = rsqrtf((row_square_sum / (float)dimension) + epsilon);
    for (dimension_index = threadIdx.x;
         dimension_index < dimension;
         dimension_index += blockDim.x)
    {
        float value;
        float weight;

        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)dimension) +
                (uint64_t)dimension_index]);
        weight = SparkGlm52ResidentDecodeStageBf16ToFloat(
            weight_bf16[dimension_index]);
        output_bf16[
            ((uint64_t)sequence_index * (uint64_t)dimension) +
            (uint64_t)dimension_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(value * inverse_rms * weight);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageBf16LinearKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t output_index;
    uint32_t sequence_index;
    uint32_t input_index;
    float local_sum;
    float row_sum;

    output_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        output_index >= output_dimension)
    {
        return;
    }
    local_sum = 0.0f;
    for (input_index = threadIdx.x;
         input_index < input_dimension;
         input_index += blockDim.x)
    {
        float activation_value;
        float weight_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        weight_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            weight_bf16[
                ((uint64_t)output_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        local_sum += activation_value * weight_value;
    }
    row_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        output_bf16[
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(row_sum);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageFp8LinearKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint8_t *__restrict__ weight_fp8_e4m3,
    const float *__restrict__ weight_scale_inv_f32,
    uint16_t *__restrict__ output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t output_index;
    uint32_t sequence_index;
    uint32_t input_index;
    uint32_t input_scale_block_count;
    uint64_t scale_row_offset;
    float local_sum;
    float row_sum;

    output_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        output_index >= output_dimension)
    {
        return;
    }
    input_scale_block_count =
        (input_dimension +
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    scale_row_offset =
        (uint64_t)(output_index /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (uint64_t)input_scale_block_count;
    local_sum = 0.0f;
    for (input_index = threadIdx.x;
         input_index < input_dimension;
         input_index += blockDim.x)
    {
        float activation_value;
        float weight_value;
        float scale_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        weight_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            weight_fp8_e4m3[
                ((uint64_t)output_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        scale_value = weight_scale_inv_f32[
            scale_row_offset +
            (uint64_t)(input_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK)];
        local_sum += activation_value * weight_value * scale_value;
    }
    row_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        output_bf16[
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(row_sum);
    }
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStagePackedNibble(
    const uint8_t *payload,
    uint64_t element_index)
{
    uint8_t packed_value;

    packed_value = payload[element_index >> 1u];
    return (element_index & 1u) == 0u
        ? (uint8_t)(packed_value & 15u)
        : (uint8_t)(packed_value >> 4u);
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
    const uint8_t *weight_payload,
    const void *weight_scale,
    uint32_t weight_format,
    uint32_t scale_block_size,
    uint32_t input_dimension,
    uint32_t output_index,
    uint32_t input_index)
{
    uint32_t input_scale_block_count;
    uint64_t weight_element_index;
    uint64_t scale_index;

    input_scale_block_count =
        (input_dimension + scale_block_size - 1u) / scale_block_size;
    scale_index =
        ((uint64_t)(output_index / scale_block_size) *
         (uint64_t)input_scale_block_count) +
        (uint64_t)(input_index / scale_block_size);
    weight_element_index =
        ((uint64_t)output_index * (uint64_t)input_dimension) +
        (uint64_t)input_index;

    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
                weight_payload[weight_element_index]) *
            ((const float *)weight_scale)[scale_index];
    }

    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
    {
        return SparkGlm52ResidentDecodeStageDecodeE2m1(
                SparkGlm52ResidentDecodeStagePackedNibble(
                    weight_payload,
                    weight_element_index)) *
            SparkGlm52ResidentDecodeStageDecodeUe4m3(
                ((const uint8_t *)weight_scale)[scale_index]);
    }

    return SparkGlm52ResidentDecodeStageDecodeE2m1(
            SparkGlm52ResidentDecodeStagePackedNibble(
                weight_payload,
                weight_element_index)) *
        SparkGlm52ResidentDecodeStageDecodeE8m0(
            ((const uint8_t *)weight_scale)[scale_index]);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageQuantizedLinearBf16Kernel(
    const uint16_t *__restrict__ input_bf16,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    uint16_t *__restrict__ output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t weight_format,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t output_index;
    uint32_t sequence_index;
    uint32_t input_index;
    float local_sum;
    float row_sum;

    output_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        output_index >= output_dimension)
    {
        return;
    }

    local_sum = 0.0f;
    for (input_index = threadIdx.x;
         input_index < input_dimension;
         input_index += blockDim.x)
    {
        float activation_value;
        float weight_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        weight_value = SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
            weight_payload,
            weight_scale,
            weight_format,
            scale_block_size,
            input_dimension,
            output_index,
            input_index);
        local_sum += activation_value * weight_value;
    }

    row_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        output_bf16[
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(row_sum);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageQuantizedLinearF32Kernel(
    const uint16_t *__restrict__ input_bf16,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    float *__restrict__ output_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t weight_format,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t output_index;
    uint32_t sequence_index;
    uint32_t input_index;
    float local_sum;
    float row_sum;

    output_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        output_index >= output_dimension)
    {
        return;
    }

    local_sum = 0.0f;
    for (input_index = threadIdx.x;
         input_index < input_dimension;
         input_index += blockDim.x)
    {
        float activation_value;
        float weight_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[
                ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                (uint64_t)input_index]);
        weight_value = SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
            weight_payload,
            weight_scale,
            weight_format,
            scale_block_size,
            input_dimension,
            output_index,
            input_index);
        local_sum += activation_value * weight_value;
    }

    row_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        output_f32[
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index] = row_sum;
    }
}


static __global__ __launch_bounds__(
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS, 4)
void SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    void *__restrict__ output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t weight_format,
    uint32_t scale_block_size,
    uint32_t output_is_f32)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __shared__ __nv_bfloat16 shared_input_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K];
    __shared__ __nv_bfloat16 shared_weight_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N];
    __shared__ float shared_output_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N];
    uint32_t sequence_tile_begin;
    uint32_t output_tile_begin;
    uint32_t tile_element_index;
    uint32_t input_tile_begin;

    sequence_tile_begin = blockIdx.y *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M;
    output_tile_begin = blockIdx.x *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        __nv_bfloat16,
        nvcuda::wmma::row_major> input_fragment;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        __nv_bfloat16,
        nvcuda::wmma::col_major> weight_fragment;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        float> accumulator_fragment;

    nvcuda::wmma::fill_fragment(accumulator_fragment, 0.0f);
    for (input_tile_begin = 0u;
         input_tile_begin < input_dimension;
         input_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K)
    {
        for (tile_element_index = threadIdx.x;
             tile_element_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
             tile_element_index += blockDim.x)
        {
            uint32_t local_sequence_index;
            uint32_t local_input_index;
            uint32_t sequence_index;
            uint32_t input_index;
            float input_value;

            local_sequence_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
            local_input_index = tile_element_index -
                (local_sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
            sequence_index = sequence_tile_begin + local_sequence_index;
            input_index = input_tile_begin + local_input_index;
            input_value = 0.0f;
            if (sequence_index < active_sequence_count &&
                input_index < input_dimension)
            {
                input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                    input_bf16[
                        ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                        (uint64_t)input_index]);
            }
            shared_input_tile[tile_element_index] = __float2bfloat16(input_value);
        }

        for (tile_element_index = threadIdx.x;
             tile_element_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
             tile_element_index += blockDim.x)
        {
            uint32_t local_input_index;
            uint32_t local_output_index;
            uint32_t input_index;
            uint32_t output_index;
            float weight_value;

            local_output_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
            local_input_index = tile_element_index -
                (local_output_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
            input_index = input_tile_begin + local_input_index;
            output_index = output_tile_begin + local_output_index;
            weight_value = 0.0f;
            if (input_index < input_dimension && output_index < output_dimension)
            {
                weight_value =
                    SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
                        weight_payload,
                        weight_scale,
                        weight_format,
                        scale_block_size,
                        input_dimension,
                        output_index,
                        input_index);
            }
            shared_weight_tile[
                local_input_index +
                (local_output_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K)] =
                __float2bfloat16(weight_value);
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(
            input_fragment,
            shared_input_tile,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
        nvcuda::wmma::load_matrix_sync(
            weight_fragment,
            shared_weight_tile,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
        nvcuda::wmma::mma_sync(
            accumulator_fragment,
            input_fragment,
            weight_fragment,
            accumulator_fragment);
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        shared_output_tile,
        accumulator_fragment,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (tile_element_index = threadIdx.x;
         tile_element_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
         tile_element_index += blockDim.x)
    {
        uint32_t local_sequence_index;
        uint32_t local_output_index;
        uint32_t sequence_index;
        uint32_t output_index;
        uint64_t output_offset;
        float output_value;

        local_sequence_index = tile_element_index /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
        local_output_index = tile_element_index -
            (local_sequence_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N);
        sequence_index = sequence_tile_begin + local_sequence_index;
        output_index = output_tile_begin + local_output_index;
        if (sequence_index >= active_sequence_count ||
            output_index >= output_dimension)
        {
            continue;
        }
        output_offset =
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index;
        output_value = shared_output_tile[tile_element_index];
        if (output_is_f32 != 0u)
        {
            ((float *)output)[output_offset] = output_value;
        }
        else
        {
            ((uint16_t *)output)[output_offset] =
                SparkGlm52ResidentDecodeStageFloatToBf16(output_value);
        }
    }
#else
    (void)input_bf16;
    (void)weight_payload;
    (void)weight_scale;
    (void)output;
    (void)active_sequence_count;
    (void)input_dimension;
    (void)output_dimension;
    (void)weight_format;
    (void)scale_block_size;
    (void)output_is_f32;
#endif
}

template <uint32_t SequenceTileRows,uint32_t OutputTileColumns>
static __global__ __launch_bounds__(
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS, 4)
void SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearWideKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    void *__restrict__ output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t weight_format,
    uint32_t scale_block_size,
    uint32_t output_is_f32)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr uint32_t row_fragment_count =
        SequenceTileRows /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M;
    constexpr uint32_t column_fragment_count =
        OutputTileColumns /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
    __shared__ __nv_bfloat16 shared_input_tile[
        SequenceTileRows *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K];
    __shared__ __nv_bfloat16 shared_weight_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K *
        OutputTileColumns];
    __shared__ float shared_output_tile[
        SequenceTileRows *
        OutputTileColumns];
    uint32_t sequence_tile_begin;
    uint32_t output_tile_begin;
    uint32_t tile_element_index;
    uint32_t input_tile_begin;
    uint32_t row_fragment_index;
    uint32_t column_fragment_index;

    sequence_tile_begin = blockIdx.y * SequenceTileRows;
    output_tile_begin = blockIdx.x * OutputTileColumns;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        __nv_bfloat16,
        nvcuda::wmma::row_major> input_fragment[row_fragment_count];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        __nv_bfloat16,
        nvcuda::wmma::col_major> weight_fragment[column_fragment_count];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K,
        float> accumulator_fragment[
            row_fragment_count * column_fragment_count];

    for (row_fragment_index = 0u;
         row_fragment_index < row_fragment_count;
         ++row_fragment_index)
    {
        for (column_fragment_index = 0u;
             column_fragment_index < column_fragment_count;
             ++column_fragment_index)
        {
            nvcuda::wmma::fill_fragment(
                accumulator_fragment[
                    (row_fragment_index * column_fragment_count) +
                    column_fragment_index],
                0.0f);
        }
    }
    for (input_tile_begin = 0u;
         input_tile_begin < input_dimension;
         input_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K)
    {
        for (tile_element_index = threadIdx.x;
             tile_element_index <
                SequenceTileRows *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
             tile_element_index += blockDim.x)
        {
            uint32_t local_sequence_index;
            uint32_t local_input_index;
            uint32_t sequence_index;
            uint32_t input_index;
            float input_value;

            local_sequence_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
            local_input_index = tile_element_index -
                (local_sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
            sequence_index = sequence_tile_begin + local_sequence_index;
            input_index = input_tile_begin + local_input_index;
            input_value = 0.0f;
            if (sequence_index < active_sequence_count &&
                input_index < input_dimension)
            {
                input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                    input_bf16[
                        ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                        (uint64_t)input_index]);
            }
            shared_input_tile[tile_element_index] = __float2bfloat16(input_value);
        }

        for (tile_element_index = threadIdx.x;
             tile_element_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K *
                OutputTileColumns;
             tile_element_index += blockDim.x)
        {
            uint32_t local_input_index;
            uint32_t local_output_index;
            uint32_t input_index;
            uint32_t output_index;
            float weight_value;

            local_output_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K;
            local_input_index = tile_element_index -
                (local_output_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
            input_index = input_tile_begin + local_input_index;
            output_index = output_tile_begin + local_output_index;
            weight_value = 0.0f;
            if (input_index < input_dimension && output_index < output_dimension)
            {
                weight_value =
                    SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
                        weight_payload,
                        weight_scale,
                        weight_format,
                        scale_block_size,
                        input_dimension,
                        output_index,
                        input_index);
            }
            shared_weight_tile[
                local_input_index +
                (local_output_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K)] =
                __float2bfloat16(weight_value);
        }
        __syncthreads();

        for (column_fragment_index = 0u;
             column_fragment_index < column_fragment_count;
             ++column_fragment_index)
        {
            nvcuda::wmma::load_matrix_sync(
                weight_fragment[column_fragment_index],
                &shared_weight_tile[
                    column_fragment_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K],
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
        }
        for (row_fragment_index = 0u;
             row_fragment_index < row_fragment_count;
             ++row_fragment_index)
        {
            nvcuda::wmma::load_matrix_sync(
                input_fragment[row_fragment_index],
                &shared_input_tile[
                    row_fragment_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K],
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K);
            for (column_fragment_index = 0u;
                 column_fragment_index < column_fragment_count;
                 ++column_fragment_index)
            {
                nvcuda::wmma::mma_sync(
                    accumulator_fragment[
                        (row_fragment_index * column_fragment_count) +
                        column_fragment_index],
                    input_fragment[row_fragment_index],
                    weight_fragment[column_fragment_index],
                    accumulator_fragment[
                        (row_fragment_index * column_fragment_count) +
                        column_fragment_index]);
            }
        }
        __syncthreads();
    }

    for (row_fragment_index = 0u;
         row_fragment_index < row_fragment_count;
         ++row_fragment_index)
    {
        for (column_fragment_index = 0u;
             column_fragment_index < column_fragment_count;
             ++column_fragment_index)
        {
            nvcuda::wmma::store_matrix_sync(
                &shared_output_tile[
                    (row_fragment_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                     OutputTileColumns) +
                    (column_fragment_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N)],
                accumulator_fragment[
                    (row_fragment_index * column_fragment_count) +
                    column_fragment_index],
                OutputTileColumns,
                nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (tile_element_index = threadIdx.x;
         tile_element_index <
            SequenceTileRows *
            OutputTileColumns;
         tile_element_index += blockDim.x)
    {
        uint32_t local_sequence_index;
        uint32_t local_output_index;
        uint32_t sequence_index;
        uint32_t output_index;
        uint64_t output_offset;
        float output_value;

        local_sequence_index = tile_element_index /
            OutputTileColumns;
        local_output_index = tile_element_index -
            (local_sequence_index * OutputTileColumns);
        sequence_index = sequence_tile_begin + local_sequence_index;
        output_index = output_tile_begin + local_output_index;
        if (sequence_index >= active_sequence_count ||
            output_index >= output_dimension)
        {
            continue;
        }
        output_offset =
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index;
        output_value = shared_output_tile[tile_element_index];
        if (output_is_f32 != 0u)
        {
            ((float *)output)[output_offset] = output_value;
        }
        else
        {
            ((uint16_t *)output)[output_offset] =
                SparkGlm52ResidentDecodeStageFloatToBf16(output_value);
        }
    }
#else
    (void)input_bf16;
    (void)weight_payload;
    (void)weight_scale;
    (void)output;
    (void)active_sequence_count;
    (void)input_dimension;
    (void)output_dimension;
    (void)weight_format;
    (void)scale_block_size;
    (void)output_is_f32;
#endif
}


static __host__ __device__ __forceinline__ uint64_t SparkGlm52ResidentDecodeStageAlignUpU64(
    uint64_t value,
    uint64_t alignment)
{
    return (value + alignment - 1ull) & ~(alignment - 1ull);
}

static __host__ __device__ __forceinline__ uint64_t SparkGlm52ResidentDecodeStageNativeActivationScaleBlockSize(
    uint32_t weight_format)
{
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K;
    }
    if (weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1 ||
        weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    return 0u;
}

static uint64_t SparkGlm52ResidentDecodeStageNativeActivationPayloadBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    uint32_t weight_format)
{
    uint64_t element_count;

    element_count =
        (uint64_t)linear_plan->maximum_active_sequence_count *
        (uint64_t)linear_plan->input_dimension;
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return element_count;
    }
    return SparkGlm52ResidentDecodeStageAlignUpU64(element_count, 2u) / 2u;
}

static uint64_t SparkGlm52ResidentDecodeStageNativeActivationScaleBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    uint32_t weight_format)
{
    uint64_t scale_block_size;
    uint64_t scale_block_count;

    scale_block_size =
        SparkGlm52ResidentDecodeStageNativeActivationScaleBlockSize(weight_format);
    if (scale_block_size == 0u)
    {
        return 0u;
    }
    scale_block_count =
        ((uint64_t)linear_plan->input_dimension + scale_block_size - 1ull) /
        scale_block_size;
    return (uint64_t)linear_plan->maximum_active_sequence_count *
        scale_block_count;
}

static uint64_t SparkGlm52ResidentDecodeStageBlackwellNativeWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    uint32_t weight_format)
{
    uint64_t payload_bytes;
    uint64_t scale_bytes;

    if (linear_plan == 0)
    {
        return 0u;
    }
    payload_bytes = SparkGlm52ResidentDecodeStageNativeActivationPayloadBytes(
        linear_plan,
        weight_format);
    scale_bytes = SparkGlm52ResidentDecodeStageNativeActivationScaleBytes(
        linear_plan,
        weight_format);
    return SparkGlm52ResidentDecodeStageAlignUpU64(
            payload_bytes,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_WORKSPACE_ALIGNMENT) +
        scale_bytes;
}

static __device__ __forceinline__ void SparkGlm52ResidentDecodeStagePackNativeFp8AFragment(
    const uint8_t *__restrict__ activation_payload,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t sequence_tile_begin,
    uint32_t input_tile_begin,
    uint32_t lane_index,
    uint32_t fragment[4])
{
    uint32_t element_index;

    fragment[0] = 0u;
    fragment[1] = 0u;
    fragment[2] = 0u;
    fragment[3] = 0u;
    for (element_index = 0u; element_index < 16u; ++element_index)
    {
        uint32_t row_in_tile;
        uint32_t input_in_tile;
        uint32_t global_sequence_index;
        uint32_t global_input_index;
        uint32_t register_index;
        uint32_t byte_index;
        uint8_t value;

        row_in_tile = ((lane_index & 3u) * 2u) + (element_index >> 3u) +
            (((lane_index >> 2u) & 1u) * 8u);
        input_in_tile = ((lane_index >> 3u) * 8u) + (element_index & 7u);
        global_sequence_index = sequence_tile_begin + row_in_tile;
        global_input_index = input_tile_begin + input_in_tile;
        value = 0u;
        if (global_sequence_index < active_sequence_count &&
            global_input_index < input_dimension)
        {
            value = activation_payload[
                ((uint64_t)global_sequence_index * (uint64_t)input_dimension) +
                (uint64_t)global_input_index];
        }
        register_index = element_index >> 2u;
        byte_index = element_index & 3u;
        fragment[register_index] |= ((uint32_t)value) << (byte_index * 8u);
    }
}

static __device__ __forceinline__ void SparkGlm52ResidentDecodeStagePackNativeFp8BFragment(
    const uint8_t *__restrict__ weight_payload,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_tile_begin,
    uint32_t input_tile_begin,
    uint32_t lane_index,
    uint32_t fragment[2])
{
    uint32_t element_index;

    fragment[0] = 0u;
    fragment[1] = 0u;
    for (element_index = 0u; element_index < 8u; ++element_index)
    {
        uint32_t input_in_tile;
        uint32_t output_in_tile;
        uint32_t global_input_index;
        uint32_t global_output_index;
        uint32_t register_index;
        uint32_t byte_index;
        uint8_t value;

        input_in_tile = ((lane_index & 15u) * 2u) + (element_index >> 2u);
        output_in_tile = ((lane_index >> 4u) * 4u) + (element_index & 3u);
        global_input_index = input_tile_begin + input_in_tile;
        global_output_index = output_tile_begin + output_in_tile;
        value = 0u;
        if (global_input_index < input_dimension &&
            global_output_index < output_dimension)
        {
            value = weight_payload[
                ((uint64_t)global_output_index * (uint64_t)input_dimension) +
                (uint64_t)global_input_index];
        }
        register_index = element_index >> 2u;
        byte_index = element_index & 3u;
        fragment[register_index] |= ((uint32_t)value) << (byte_index * 8u);
    }
}

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageLoadPackedNativeFp4(
    const uint8_t *__restrict__ payload,
    uint64_t element_index)
{
    uint8_t packed_value;

    packed_value = payload[element_index >> 1u];
    return (element_index & 1u) == 0u
        ? (uint8_t)(packed_value & 15u)
        : (uint8_t)(packed_value >> 4u);
}

static __device__ __forceinline__ void SparkGlm52ResidentDecodeStagePackNativeFp4AFragment(
    const uint8_t *__restrict__ activation_payload,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t sequence_tile_begin,
    uint32_t input_tile_begin,
    uint32_t lane_index,
    uint32_t fragment[4])
{
    uint32_t element_index;

    fragment[0] = 0u;
    fragment[1] = 0u;
    fragment[2] = 0u;
    fragment[3] = 0u;
    for (element_index = 0u; element_index < 32u; ++element_index)
    {
        uint32_t row_in_tile;
        uint32_t input_in_tile;
        uint32_t global_sequence_index;
        uint32_t global_input_index;
        uint32_t register_index;
        uint32_t nibble_index;
        uint8_t value;

        row_in_tile = ((lane_index & 3u) * 2u) + ((element_index >> 4u) * 8u);
        input_in_tile = ((lane_index >> 2u) * 8u) + (element_index & 7u) +
            (((element_index >> 3u) & 1u) * 32u);
        global_sequence_index = sequence_tile_begin + row_in_tile;
        global_input_index = input_tile_begin + input_in_tile;
        value = 0u;
        if (global_sequence_index < active_sequence_count &&
            global_input_index < input_dimension)
        {
            value = SparkGlm52ResidentDecodeStageLoadPackedNativeFp4(
                activation_payload,
                ((uint64_t)global_sequence_index * (uint64_t)input_dimension) +
                    (uint64_t)global_input_index);
        }
        register_index = element_index >> 3u;
        nibble_index = element_index & 7u;
        fragment[register_index] |= ((uint32_t)value) << (nibble_index * 4u);
    }
}

static __device__ __forceinline__ void SparkGlm52ResidentDecodeStagePackNativeFp4BFragment(
    const uint8_t *__restrict__ weight_payload,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_tile_begin,
    uint32_t input_tile_begin,
    uint32_t lane_index,
    uint32_t fragment[2])
{
    uint32_t element_index;

    fragment[0] = 0u;
    fragment[1] = 0u;
    for (element_index = 0u; element_index < 16u; ++element_index)
    {
        uint32_t input_in_tile;
        uint32_t output_in_tile;
        uint32_t global_input_index;
        uint32_t global_output_index;
        uint32_t register_index;
        uint32_t nibble_index;
        uint8_t value;

        input_in_tile = ((lane_index & 15u) * 4u) + (element_index >> 2u);
        output_in_tile = ((lane_index >> 4u) * 4u) + (element_index & 3u);
        global_input_index = input_tile_begin + input_in_tile;
        global_output_index = output_tile_begin + output_in_tile;
        value = 0u;
        if (global_input_index < input_dimension &&
            global_output_index < output_dimension)
        {
            value = SparkGlm52ResidentDecodeStageLoadPackedNativeFp4(
                weight_payload,
                ((uint64_t)global_output_index * (uint64_t)input_dimension) +
                    (uint64_t)global_input_index);
        }
        register_index = element_index >> 3u;
        nibble_index = element_index & 7u;
        fragment[register_index] |= ((uint32_t)value) << (nibble_index * 4u);
    }
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageLoadNativeActivationScaleData(
    const uint8_t *__restrict__ activation_scale,
    uint32_t sequence_index,
    uint32_t input_scale_block_count,
    uint32_t first_scale_block_index,
    uint32_t scale_count)
{
    uint32_t scale_data;
    uint32_t scale_index;

    scale_data = 0u;
    for (scale_index = 0u; scale_index < scale_count; ++scale_index)
    {
        scale_data |= ((uint32_t)activation_scale[
            ((uint64_t)sequence_index * (uint64_t)input_scale_block_count) +
            (uint64_t)(first_scale_block_index + scale_index)]) <<
            (scale_index * 8u);
    }
    return scale_data;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageLoadNativeWeightScaleData(
    const void *__restrict__ weight_scale,
    uint32_t weight_format,
    uint32_t scale_block_size,
    uint32_t input_dimension,
    uint32_t output_index,
    uint32_t first_scale_block_index,
    uint32_t scale_count)
{
    uint32_t input_scale_block_count;
    uint32_t scale_data;
    uint32_t scale_index;

    input_scale_block_count = (input_dimension + scale_block_size - 1u) /
        scale_block_size;
    scale_data = 0u;
    for (scale_index = 0u; scale_index < scale_count; ++scale_index)
    {
        uint64_t source_scale_index;
        uint8_t encoded_scale;

        source_scale_index =
            ((uint64_t)(output_index / scale_block_size) *
             (uint64_t)input_scale_block_count) +
            (uint64_t)(first_scale_block_index + scale_index);
        if (weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
        {
            encoded_scale = SparkGlm52ResidentDecodeStageEncodeE8m0Saturate(
                ((const float *)weight_scale)[source_scale_index]);
        }
        else
        {
            encoded_scale = ((const uint8_t *)weight_scale)[source_scale_index];
        }
        scale_data |= ((uint32_t)encoded_scale) << (scale_index * 8u);
    }
    return scale_data;
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageBlackwellNativeQuantizeActivationKernel(
    const uint16_t *__restrict__ input_bf16,
    uint8_t *__restrict__ activation_payload,
    uint8_t *__restrict__ activation_scale,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t weight_format)
{
    __shared__ float shared_reduction[256u];
    uint32_t sequence_index;
    uint32_t scale_block_index;
    uint32_t scale_block_size;
    uint32_t input_begin;
    uint32_t input_end;
    uint32_t input_index;
    float local_absmax;
    float block_absmax;
    float scale_value;
    uint8_t encoded_scale;

    sequence_index = blockIdx.x;
    scale_block_index = blockIdx.y;
    scale_block_size = (uint32_t)
        SparkGlm52ResidentDecodeStageNativeActivationScaleBlockSize(weight_format);
    if (sequence_index >= active_sequence_count || scale_block_size == 0u)
    {
        return;
    }

    input_begin = scale_block_index * scale_block_size;
    input_end = input_begin + scale_block_size;
    if (input_begin >= input_dimension)
    {
        return;
    }
    if (input_end > input_dimension)
    {
        input_end = input_dimension;
    }

    local_absmax = 0.0f;
    for (input_index = input_begin + threadIdx.x;
         input_index < input_end;
         input_index += blockDim.x)
    {
        float value;

        value = fabsf(SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)sequence_index * input_dimension) +
                input_index]));
        local_absmax = fmaxf(local_absmax, value);
    }
    block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
        local_absmax,
        shared_reduction);

    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
        encoded_scale = SparkGlm52ResidentDecodeStageEncodeE8m0Saturate(scale_value);
    }
    else
    {
        scale_value = fmaxf(block_absmax / 6.0f, 1.0e-8f);
        encoded_scale = weight_format ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1
            ? SparkGlm52ResidentDecodeStageEncodeUe4m3Saturate(scale_value)
            : SparkGlm52ResidentDecodeStageEncodeE8m0Saturate(scale_value);
    }

    if (threadIdx.x == 0u)
    {
        uint32_t input_scale_block_count;

        input_scale_block_count = (input_dimension + scale_block_size - 1u) /
            scale_block_size;
        activation_scale[
            ((uint64_t)sequence_index * (uint64_t)input_scale_block_count) +
            (uint64_t)scale_block_index] = encoded_scale;
    }
    __syncthreads();

    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        for (input_index = input_begin + threadIdx.x;
             input_index < input_end;
             input_index += blockDim.x)
        {
            float input_value;

            input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[((uint64_t)sequence_index * input_dimension) +
                    input_index]) / scale_value;
            activation_payload[((uint64_t)sequence_index * input_dimension) +
                input_index] =
                SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(input_value);
        }
    }
    else
    {
        for (input_index = input_begin + threadIdx.x;
             input_index < input_end;
             input_index += blockDim.x)
        {
            float input_value;
            uint8_t packed_value;
            uint8_t quantized_value;
            uint64_t payload_index;

            input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[((uint64_t)sequence_index * input_dimension) +
                    input_index]) / scale_value;
            quantized_value =
                SparkGlm52ResidentDecodeStageEncodeE2m1Saturate(input_value);
            payload_index = (((uint64_t)sequence_index * input_dimension) +
                input_index) >> 1u;
            packed_value = activation_payload[payload_index];
            if ((input_index & 1u) == 0u)
            {
                packed_value = (uint8_t)((packed_value & 0xf0u) |
                    (quantized_value & 15u));
            }
            else
            {
                packed_value = (uint8_t)((packed_value & 15u) |
                    (uint8_t)(quantized_value << 4u));
            }
            activation_payload[payload_index] = packed_value;
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_THREADS, 2)
void SparkGlm52ResidentDecodeStageBlackwellNativeFp8TensorCoreLinearKernel(
    const uint8_t *__restrict__ activation_payload,
    const uint8_t *__restrict__ activation_scale,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    void *__restrict__ output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t output_is_f32)
{
#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_NATIVE_BLOCK_SCALED_MMA && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t sequence_tile_begin;
    uint32_t output_tile_begin;
    uint32_t input_scale_block_count;
    float accumulator[4];

    lane_index = threadIdx.x & 31u;
    warp_index = threadIdx.x >> 5u;
    sequence_tile_begin = (blockIdx.y * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_SEQUENCE_GROUP) +
        (warp_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_M);
    output_tile_begin = blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_N;
    input_scale_block_count =
        (input_dimension + SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K;
    accumulator[0] = 0.0f;
    accumulator[1] = 0.0f;
    accumulator[2] = 0.0f;
    accumulator[3] = 0.0f;

    for (uint32_t input_tile_begin = 0u;
         input_tile_begin < input_dimension;
         input_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K)
    {
        uint32_t a_fragment[4];
        uint32_t b_fragment[2];
        uint32_t scale_a_data;
        uint32_t scale_b_data;
        uint16_t scale_byte_id;
        uint16_t scale_thread_id;

        SparkGlm52ResidentDecodeStagePackNativeFp8AFragment(
            activation_payload,
            active_sequence_count,
            input_dimension,
            sequence_tile_begin,
            input_tile_begin,
            lane_index,
            a_fragment);
        SparkGlm52ResidentDecodeStagePackNativeFp8BFragment(
            weight_payload,
            input_dimension,
            output_dimension,
            output_tile_begin,
            input_tile_begin,
            lane_index,
            b_fragment);
        scale_a_data = SparkGlm52ResidentDecodeStageLoadNativeActivationScaleData(
            activation_scale,
            sequence_tile_begin + ((lane_index & 3u) * 2u),
            input_scale_block_count,
            input_tile_begin / SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP8_MMA_K,
            1u);
        scale_b_data = SparkGlm52ResidentDecodeStageLoadNativeWeightScaleData(
            weight_scale,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK,
            input_dimension,
            output_tile_begin + (lane_index >> 4u),
            input_tile_begin / SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK,
            1u);
        scale_byte_id = 0u;
        scale_thread_id = 0u;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3}, "
            "%10, {%12, %13}, %11, {%12, %13};\n"
            : "+f"(accumulator[0]), "+f"(accumulator[1]),
              "+f"(accumulator[2]), "+f"(accumulator[3])
            : "r"(a_fragment[0]), "r"(a_fragment[1]),
              "r"(a_fragment[2]), "r"(a_fragment[3]),
              "r"(b_fragment[0]), "r"(b_fragment[1]),
              "r"(scale_a_data), "r"(scale_b_data),
              "h"(scale_byte_id), "h"(scale_thread_id));
    }

    for (uint32_t accumulator_index = 0u; accumulator_index < 4u; ++accumulator_index)
    {
        uint32_t row_offset;
        uint32_t column_offset;
        uint32_t sequence_index;
        uint32_t output_index;
        uint64_t output_offset;

        row_offset = ((accumulator_index & 1u) * 8u) + (lane_index >> 2u);
        column_offset = ((lane_index & 3u) * 2u) + (accumulator_index >> 1u);
        sequence_index = sequence_tile_begin + row_offset;
        output_index = output_tile_begin + column_offset;
        if (sequence_index >= active_sequence_count || output_index >= output_dimension)
        {
            continue;
        }
        output_offset =
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index;
        if (output_is_f32 != 0u)
        {
            ((float *)output)[output_offset] = accumulator[accumulator_index];
        }
        else
        {
            ((uint16_t *)output)[output_offset] =
                SparkGlm52ResidentDecodeStageFloatToBf16(accumulator[accumulator_index]);
        }
    }
#else
    (void)activation_payload;
    (void)activation_scale;
    (void)weight_payload;
    (void)weight_scale;
    (void)output;
    (void)active_sequence_count;
    (void)input_dimension;
    (void)output_dimension;
    (void)output_is_f32;
#endif
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_THREADS, 2)
void SparkGlm52ResidentDecodeStageBlackwellNativeFp4TensorCoreLinearKernel(
    const uint8_t *__restrict__ activation_payload,
    const uint8_t *__restrict__ activation_scale,
    const uint8_t *__restrict__ weight_payload,
    const void *__restrict__ weight_scale,
    void *__restrict__ output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t weight_format,
    uint32_t scale_block_size,
    uint32_t output_is_f32)
{
#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_NATIVE_BLOCK_SCALED_MMA && \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t sequence_tile_begin;
    uint32_t output_tile_begin;
    uint32_t input_activation_scale_block_count;
    float accumulator[4];

    lane_index = threadIdx.x & 31u;
    warp_index = threadIdx.x >> 5u;
    sequence_tile_begin = (blockIdx.y * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_SEQUENCE_GROUP) +
        (warp_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_M);
    output_tile_begin = blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_MMA_N;
    input_activation_scale_block_count =
        (input_dimension + SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE - 1u) /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    accumulator[0] = 0.0f;
    accumulator[1] = 0.0f;
    accumulator[2] = 0.0f;
    accumulator[3] = 0.0f;

    for (uint32_t input_tile_begin = 0u;
         input_tile_begin < input_dimension;
         input_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_NATIVE_FP4_MMA_K)
    {
        uint32_t a_fragment[4];
        uint32_t b_fragment[2];
        uint32_t scale_a_data;
        uint32_t scale_b_data;
        uint16_t scale_byte_id;
        uint16_t scale_thread_id;

        SparkGlm52ResidentDecodeStagePackNativeFp4AFragment(
            activation_payload,
            active_sequence_count,
            input_dimension,
            sequence_tile_begin,
            input_tile_begin,
            lane_index,
            a_fragment);
        SparkGlm52ResidentDecodeStagePackNativeFp4BFragment(
            weight_payload,
            input_dimension,
            output_dimension,
            output_tile_begin,
            input_tile_begin,
            lane_index,
            b_fragment);
        scale_a_data = SparkGlm52ResidentDecodeStageLoadNativeActivationScaleData(
            activation_scale,
            sequence_tile_begin + ((lane_index & 3u) * 2u),
            input_activation_scale_block_count,
            input_tile_begin / SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            4u);
        scale_b_data = SparkGlm52ResidentDecodeStageLoadNativeWeightScaleData(
            weight_scale,
            weight_format,
            scale_block_size,
            input_dimension,
            output_tile_begin + (lane_index >> 4u),
            input_tile_begin / scale_block_size,
            scale_block_size == SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE ? 4u : 2u);
        scale_byte_id = 0u;
        scale_thread_id = 0u;
        if (weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
        {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3}, "
                "%10, {%12, %13}, %11, {%12, %13};\n"
                : "+f"(accumulator[0]), "+f"(accumulator[1]),
                  "+f"(accumulator[2]), "+f"(accumulator[3])
                : "r"(a_fragment[0]), "r"(a_fragment[1]),
                  "r"(a_fragment[2]), "r"(a_fragment[3]),
                  "r"(b_fragment[0]), "r"(b_fragment[1]),
                  "r"(scale_a_data), "r"(scale_b_data),
                  "h"(scale_byte_id), "h"(scale_thread_id));
        }
        else
        {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3}, "
                "%10, {%12, %13}, %11, {%12, %13};\n"
                : "+f"(accumulator[0]), "+f"(accumulator[1]),
                  "+f"(accumulator[2]), "+f"(accumulator[3])
                : "r"(a_fragment[0]), "r"(a_fragment[1]),
                  "r"(a_fragment[2]), "r"(a_fragment[3]),
                  "r"(b_fragment[0]), "r"(b_fragment[1]),
                  "r"(scale_a_data), "r"(scale_b_data),
                  "h"(scale_byte_id), "h"(scale_thread_id));
        }
    }

    for (uint32_t accumulator_index = 0u; accumulator_index < 4u; ++accumulator_index)
    {
        uint32_t row_offset;
        uint32_t column_offset;
        uint32_t sequence_index;
        uint32_t output_index;
        uint64_t output_offset;

        row_offset = ((accumulator_index & 1u) * 8u) + (lane_index >> 2u);
        column_offset = ((lane_index & 3u) * 2u) + (accumulator_index >> 1u);
        sequence_index = sequence_tile_begin + row_offset;
        output_index = output_tile_begin + column_offset;
        if (sequence_index >= active_sequence_count || output_index >= output_dimension)
        {
            continue;
        }
        output_offset =
            ((uint64_t)sequence_index * (uint64_t)output_dimension) +
            (uint64_t)output_index;
        if (output_is_f32 != 0u)
        {
            ((float *)output)[output_offset] = accumulator[accumulator_index];
        }
        else
        {
            ((uint16_t *)output)[output_offset] =
                SparkGlm52ResidentDecodeStageFloatToBf16(accumulator[accumulator_index]);
        }
    }
#else
    (void)activation_payload;
    (void)activation_scale;
    (void)weight_payload;
    (void)weight_scale;
    (void)output;
    (void)active_sequence_count;
    (void)input_dimension;
    (void)output_dimension;
    (void)weight_format;
    (void)scale_block_size;
    (void)output_is_f32;
#endif
}

static __global__ void SparkGlm52ResidentDecodeStageDsaSelectKernel(
    const float *__restrict__ dsa_token_scores,
    const uint32_t *__restrict__ context_lengths,
    uint32_t *__restrict__ sparse_token_indices,
    uint32_t active_sequence_count,
    uint32_t dsa_candidate_count)
{
    uint32_t sequence_index;
    uint32_t selected_index;
    uint64_t row_offset;

    sequence_index = blockIdx.x;
    if (threadIdx.x != 0u || sequence_index >= active_sequence_count)
    {
        return;
    }
    row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    for (selected_index = 0u;
         selected_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         ++selected_index)
    {
        uint32_t candidate_index;
        uint32_t best_token_index;
        float best_score;

        best_token_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
        best_score = -FLT_MAX;
        for (candidate_index = 0u;
             candidate_index < dsa_candidate_count;
             ++candidate_index)
        {
            uint32_t prior_index;
            uint32_t already_selected;
            float candidate_score;

            already_selected = 0u;
            for (prior_index = 0u; prior_index < selected_index; ++prior_index)
            {
                if (sparse_token_indices[row_offset + prior_index] ==
                    candidate_index)
                {
                    already_selected = 1u;
                }
            }
            if (already_selected != 0u ||
                candidate_index >= context_lengths[sequence_index])
            {
                continue;
            }
            candidate_score = dsa_token_scores[
                ((uint64_t)sequence_index * (uint64_t)dsa_candidate_count) +
                (uint64_t)candidate_index];
            if (candidate_score > best_score ||
                (candidate_score == best_score && candidate_index < best_token_index))
            {
                best_score = candidate_score;
                best_token_index = candidate_index;
            }
        }
        sparse_token_indices[row_offset + selected_index] = best_token_index;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageCopyContextPrefixSparseIndicesKernel(
    const uint32_t *__restrict__ context_lengths,
    uint32_t *__restrict__ sparse_token_indices,
    uint32_t active_sequence_count)
{
    uint32_t sequence_index;
    uint32_t selected_index;
    uint32_t context_length;
    uint64_t row_offset;

    sequence_index = blockIdx.x;
    selected_index = threadIdx.x;
    if (sequence_index >= active_sequence_count)
    {
        return;
    }

    context_length = context_lengths[sequence_index];
    row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    while (selected_index <
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT)
    {
        sparse_token_indices[row_offset + (uint64_t)selected_index] =
            selected_index < context_length
            ? selected_index
            : SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
        selected_index += blockDim.x;
    }
}

static __global__ void SparkGlm52ResidentDecodeStagePrepareKernel(
    const uint16_t *__restrict__ query_rope_input_bf16,
    const uint16_t *__restrict__ key_rope_input_bf16,
    const uint16_t *__restrict__ current_kv_latent_bf16,
    const uint16_t *__restrict__ raw_kv_b_bf16,
    const uint32_t *__restrict__ positions,
    const uint32_t *__restrict__ slot_mapping,
    const float *__restrict__ cos_table,
    const float *__restrict__ sin_table,
    uint16_t *__restrict__ rotated_query_rope_bf16,
    uint16_t *__restrict__ mla_cache_bf16,
    uint16_t *__restrict__ key_nope_cache_bf16,
    uint16_t *__restrict__ value_cache_bf16,
    uint32_t active_sequence_count,
    uint32_t position_count,
    uint32_t cache_token_capacity)
{
    const uint32_t rope_pair_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
    uint64_t query_rope_work_count;
    uint64_t cache_latent_work_count;
    uint64_t cache_rope_work_count;
    uint64_t cache_key_nope_work_count;
    uint64_t cache_value_work_count;
    uint64_t total_work_count;
    uint64_t work_index;
    uint64_t work_stride;

    query_rope_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)rope_pair_count;
    cache_latent_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    cache_rope_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)rope_pair_count;
    cache_key_nope_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
    cache_value_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
    total_work_count =
        query_rope_work_count + cache_latent_work_count +
        cache_rope_work_count + cache_key_nope_work_count +
        cache_value_work_count;
    work_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    work_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    while (work_index < total_work_count)
    {
        if (work_index < query_rope_work_count)
        {
            uint64_t query_row_index;
            uint32_t rope_pair_index;
            uint32_t sequence_index;
            uint32_t position;
            uint64_t input_offset;
            uint64_t table_offset;

            query_row_index = work_index / (uint64_t)rope_pair_count;
            rope_pair_index =
                (uint32_t)(work_index % (uint64_t)rope_pair_count);
            sequence_index = (uint32_t)(
                query_row_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
            position = positions[sequence_index];
            input_offset =
                (query_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                ((uint64_t)rope_pair_index * 2u);
            if (position < position_count)
            {
                table_offset =
                    ((uint64_t)position * (uint64_t)rope_pair_count) +
                    (uint64_t)rope_pair_index;
                SparkGlm52ResidentDecodeStageApplyRopePair(
                    query_rope_input_bf16[input_offset],
                    query_rope_input_bf16[input_offset + 1u],
                    cos_table[table_offset],
                    sin_table[table_offset],
                    &rotated_query_rope_bf16[input_offset],
                    &rotated_query_rope_bf16[input_offset + 1u]);
            }
            else
            {
                rotated_query_rope_bf16[input_offset] = 0u;
                rotated_query_rope_bf16[input_offset + 1u] = 0u;
            }
        }
        else if (work_index <
                 query_rope_work_count + cache_latent_work_count)
        {
            uint64_t cache_work_index;
            uint32_t sequence_index;
            uint32_t latent_dimension_index;
            uint32_t cache_slot_index;
            uint64_t cache_offset;

            cache_work_index = work_index - query_rope_work_count;
            sequence_index = (uint32_t)(
                cache_work_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
            latent_dimension_index = (uint32_t)(
                cache_work_index %
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
            cache_slot_index = slot_mapping[sequence_index];
            if (cache_slot_index < cache_token_capacity)
            {
                cache_offset =
                    ((uint64_t)cache_slot_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    (uint64_t)latent_dimension_index;
                mla_cache_bf16[cache_offset] = current_kv_latent_bf16[
                    ((uint64_t)sequence_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)latent_dimension_index];
            }
        }
        else if (work_index <
                 query_rope_work_count + cache_latent_work_count +
                 cache_rope_work_count)
        {
            uint64_t cache_rope_work_index;
            uint32_t sequence_index;
            uint32_t rope_pair_index;
            uint32_t cache_slot_index;
            uint32_t position;
            uint64_t input_offset;
            uint64_t cache_offset;
            uint64_t table_offset;

            cache_rope_work_index =
                work_index - query_rope_work_count - cache_latent_work_count;
            sequence_index = (uint32_t)(
                cache_rope_work_index / (uint64_t)rope_pair_count);
            rope_pair_index = (uint32_t)(
                cache_rope_work_index % (uint64_t)rope_pair_count);
            cache_slot_index = slot_mapping[sequence_index];
            position = positions[sequence_index];
            if (cache_slot_index < cache_token_capacity &&
                position < position_count)
            {
                input_offset =
                    ((uint64_t)sequence_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                    ((uint64_t)rope_pair_index * 2u);
                cache_offset =
                    ((uint64_t)cache_slot_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                    ((uint64_t)rope_pair_index * 2u);
                table_offset =
                    ((uint64_t)position * (uint64_t)rope_pair_count) +
                    (uint64_t)rope_pair_index;
                SparkGlm52ResidentDecodeStageApplyRopePair(
                    key_rope_input_bf16[input_offset],
                    key_rope_input_bf16[input_offset + 1u],
                    cos_table[table_offset],
                    sin_table[table_offset],
                    &mla_cache_bf16[cache_offset],
                    &mla_cache_bf16[cache_offset + 1u]);
            }
        }
        if (work_index >= query_rope_work_count + cache_latent_work_count +
                cache_rope_work_count &&
            work_index < query_rope_work_count + cache_latent_work_count +
                cache_rope_work_count + cache_key_nope_work_count)
        {
            uint64_t cache_key_work_index;
            uint64_t row_index;
            uint32_t sequence_index;
            uint32_t head_index;
            uint32_t dimension_index;
            uint32_t cache_slot_index;
            uint64_t raw_offset;
            uint64_t cache_offset;

            cache_key_work_index =
                work_index - query_rope_work_count - cache_latent_work_count -
                cache_rope_work_count;
            dimension_index = (uint32_t)(
                cache_key_work_index %
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION);
            row_index =
                cache_key_work_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
            head_index = (uint32_t)(
                row_index %
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
            sequence_index = (uint32_t)(
                row_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
            cache_slot_index = slot_mapping[sequence_index];
            if (raw_kv_b_bf16 != 0 && cache_slot_index < cache_token_capacity)
            {
                raw_offset =
                    (((uint64_t)sequence_index *
                      (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                      (uint64_t)head_index) *
                     (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
                    (uint64_t)dimension_index;
                cache_offset =
                    (((uint64_t)cache_slot_index *
                      (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                      (uint64_t)head_index) *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                    (uint64_t)dimension_index;
                key_nope_cache_bf16[cache_offset] = raw_kv_b_bf16[raw_offset];
            }
        }
        else if (work_index >= query_rope_work_count + cache_latent_work_count +
                     cache_rope_work_count + cache_key_nope_work_count)
        {
            uint64_t cache_value_work_index;
            uint64_t row_index;
            uint32_t sequence_index;
            uint32_t head_index;
            uint32_t dimension_index;
            uint32_t cache_slot_index;
            uint64_t raw_offset;
            uint64_t cache_offset;

            cache_value_work_index =
                work_index - query_rope_work_count - cache_latent_work_count -
                cache_rope_work_count - cache_key_nope_work_count;
            dimension_index = (uint32_t)(
                cache_value_work_index %
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION);
            row_index =
                cache_value_work_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
            head_index = (uint32_t)(
                row_index %
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
            sequence_index = (uint32_t)(
                row_index /
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT);
            cache_slot_index = slot_mapping[sequence_index];
            if (raw_kv_b_bf16 != 0 && cache_slot_index < cache_token_capacity)
            {
                raw_offset =
                    (((uint64_t)sequence_index *
                      (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                      (uint64_t)head_index) *
                     (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                    (uint64_t)dimension_index;
                cache_offset =
                    (((uint64_t)cache_slot_index *
                      (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                      (uint64_t)head_index) *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                    (uint64_t)dimension_index;
                value_cache_bf16[cache_offset] = raw_kv_b_bf16[raw_offset];
            }
        }
        work_index += work_stride;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageMapRawQueryKernel(
    const uint16_t *__restrict__ raw_query_b_bf16,
    uint16_t *__restrict__ query_latent_bf16,
    uint16_t *__restrict__ query_rope_input_bf16,
    uint32_t active_sequence_count)
{
    uint64_t total_work_count;
    uint64_t work_index;
    uint64_t work_stride;

    total_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    work_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    work_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    while (work_index < total_work_count)
    {
        uint32_t latent_dimension_index;
        uint64_t row_index;
        uint64_t raw_offset;
        uint16_t value_bf16;

        latent_dimension_index = (uint32_t)(
            work_index %
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
        row_index =
            work_index /
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
        value_bf16 = 0u;
        if (latent_dimension_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)
        {
            raw_offset =
                (row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)latent_dimension_index;
            value_bf16 = raw_query_b_bf16[raw_offset];
        }
        query_latent_bf16[work_index] = value_bf16;
        if (latent_dimension_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION)
        {
            raw_offset =
                (row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                (uint64_t)latent_dimension_index;
            query_rope_input_bf16[
                (row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                (uint64_t)latent_dimension_index] =
                raw_query_b_bf16[raw_offset];
        }
        work_index += work_stride;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageSplitRawKvAKernel(
    const uint16_t *__restrict__ raw_kv_a_bf16,
    uint16_t *__restrict__ current_kv_latent_bf16,
    uint16_t *__restrict__ key_rope_input_bf16,
    uint32_t active_sequence_count)
{
    uint64_t total_work_count;
    uint64_t work_index;
    uint64_t work_stride;

    total_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION;
    work_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    work_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    while (work_index < total_work_count)
    {
        uint32_t kv_dimension_index;
        uint32_t sequence_index;

        kv_dimension_index = (uint32_t)(
            work_index %
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
        sequence_index = (uint32_t)(
            work_index /
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION);
        if (kv_dimension_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)
        {
            current_kv_latent_bf16[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                (uint64_t)kv_dimension_index] = raw_kv_a_bf16[work_index];
        }
        else
        {
            key_rope_input_bf16[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                (uint64_t)(kv_dimension_index -
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)] =
                raw_kv_a_bf16[work_index];
        }
        work_index += work_stride;
    }
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageResolveCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity)
{
    uint64_t addressed_token_index;
    uint64_t logical_block_index;
    uint64_t block_token_offset;
    uint64_t block_table_offset;
    uint32_t physical_block_index;
    uint64_t cache_slot_index;

    addressed_token_index =
        (uint64_t)first_block_token_offset + (uint64_t)token_index;
    logical_block_index =
        addressed_token_index / (uint64_t)block_token_count;
    if (logical_block_index >= (uint64_t)max_blocks_per_sequence)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    block_token_offset =
        addressed_token_index % (uint64_t)block_token_count;
    block_table_offset =
        ((uint64_t)sequence_index *
         (uint64_t)max_blocks_per_sequence) +
        logical_block_index;
    physical_block_index = block_table[block_table_offset];
    if (physical_block_index >= kv_block_count)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    cache_slot_index =
        ((uint64_t)physical_block_index * (uint64_t)block_token_count) +
        block_token_offset;
    if (cache_slot_index >= (uint64_t)cache_token_capacity ||
        cache_slot_index > (uint64_t)UINT32_MAX)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    return (uint32_t)cache_slot_index;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageWarpResolveCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    uint32_t lane_index)
{
    uint32_t cache_slot_index;

    cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    if (lane_index == 0u)
    {
        cache_slot_index = SparkGlm52ResidentDecodeStageResolveCacheSlot(
            block_table,
            sequence_index,
            token_index,
            first_block_token_offset,
            block_token_count,
            max_blocks_per_sequence,
            kv_block_count,
            cache_token_capacity);
    }
    return SparkGlm52ResidentDecodeStageWarpBroadcastU32(cache_slot_index, 0u);
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageWarpDotProduct(
    const float *shared_query,
    const uint16_t *key_nope_cache_bf16,
    const uint16_t *mla_cache_bf16,
    uint32_t cache_slot_index,
    uint32_t head_index,
    uint32_t lane_index,
    float qk_scale)
{
    const uint32_t *key_pairs;
    const uint32_t *rope_pairs;
    uint64_t key_element_offset;
    uint64_t cache_element_offset;
    float accumulated_dot_product;
    uint32_t pair_index;

    key_element_offset =
        (((uint64_t)cache_slot_index *
          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
          (uint64_t)head_index) *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION);
    cache_element_offset =
        (uint64_t)cache_slot_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS +
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    key_pairs = (const uint32_t *)&key_nope_cache_bf16[key_element_offset];
    rope_pairs = (const uint32_t *)&mla_cache_bf16[cache_element_offset];
    accumulated_dot_product = 0.0f;
    for (pair_index = lane_index;
         pair_index <
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION / 2u;
         pair_index += SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
    {
        uint32_t packed_values;
        uint32_t first_dimension;

        packed_values = key_pairs[pair_index];
        first_dimension = pair_index * 2u;
        accumulated_dot_product +=
            shared_query[first_dimension] *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                (uint16_t)(packed_values & 0xffffu));
        accumulated_dot_product +=
            shared_query[first_dimension + 1u] *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                (uint16_t)(packed_values >> 16u));
    }
    for (pair_index = lane_index;
         pair_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
         pair_index += SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
    {
        uint32_t packed_values;
        uint32_t first_dimension;

        packed_values = rope_pairs[pair_index];
        first_dimension =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            (pair_index * 2u);
        accumulated_dot_product +=
            shared_query[first_dimension] *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                (uint16_t)(packed_values & 0xffffu));
        accumulated_dot_product +=
            shared_query[first_dimension + 1u] *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                (uint16_t)(packed_values >> 16u));
    }
    return SparkGlm52ResidentDecodeStageWarpReduceSum(
        accumulated_dot_product) * qk_scale;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageAttentionKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint16_t *__restrict__ key_nope_cache_bf16,
    const uint16_t *__restrict__ value_cache_bf16,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint16_t *__restrict__ output_value_bf16,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_query[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];
    __shared__ float shared_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT];
    __shared__ uint32_t shared_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT];
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t sequence_index;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint64_t query_row_index;
    uint64_t sparse_row_offset;
    uint64_t output_row_offset;
    uint32_t dimension_index;
    uint32_t candidate_index;
    float local_maximum;
    float row_maximum;
    float local_exponential_sum;
    float row_exponential_sum;

    sequence_index = blockIdx.x;
    head_index = blockIdx.y;
    lane_index =
        threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index =
        threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    warp_count =
        blockDim.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    context_length = context_lengths[sequence_index];
    first_block_token_offset = first_block_token_offsets[sequence_index];
    query_row_index =
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
        (uint64_t)head_index;
    sparse_row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    output_row_offset =
        query_row_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;

    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[dimension_index] =
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                query_latent_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            dimension_index] = SparkGlm52ResidentDecodeStageBf16ToFloat(
                rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    __syncthreads();

    local_maximum = -FLT_MAX;
    for (candidate_index = warp_index;
         candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         candidate_index += warp_count)
    {
        uint32_t token_index;
        uint32_t cache_slot_index;
        float attention_score;

        token_index = sparse_token_indices[
            sparse_row_offset + (uint64_t)candidate_index];
        cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
        if (token_index < context_length)
        {
            cache_slot_index = SparkGlm52ResidentDecodeStageWarpResolveCacheSlot(
                block_table,
                sequence_index,
                token_index,
                first_block_token_offset,
                block_token_count,
                max_blocks_per_sequence,
                kv_block_count,
                cache_token_capacity,
                lane_index);
        }
        attention_score =
            cache_slot_index != SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT
            ? SparkGlm52ResidentDecodeStageWarpDotProduct(
                  shared_query,
                  key_nope_cache_bf16,
                  mla_cache_bf16,
                  cache_slot_index,
                  head_index,
                  lane_index,
                  qk_scale)
            : -FLT_MAX;
        if (lane_index == 0u)
        {
            shared_scores[candidate_index] = attention_score;
            shared_cache_slots[candidate_index] = cache_slot_index;
            if (attention_score > local_maximum)
            {
                local_maximum = attention_score;
            }
        }
    }

    row_maximum = SparkGlm52ResidentDecodeStageBlockReduceMax(
        local_maximum,
        shared_reduction);
    local_exponential_sum = 0.0f;
    for (candidate_index = threadIdx.x;
         candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         candidate_index += blockDim.x)
    {
        float exponential_score;

        exponential_score = row_maximum <= (-FLT_MAX * 0.5f)
            ? 0.0f
            : __expf(shared_scores[candidate_index] - row_maximum);
        shared_scores[candidate_index] = exponential_score;
        local_exponential_sum += exponential_score;
    }
    row_exponential_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_exponential_sum,
        shared_reduction);

    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
         dimension_index += blockDim.x)
    {
        float accumulated_value;

        accumulated_value = 0.0f;
        if (row_exponential_sum > 0.0f)
        {
            for (candidate_index = 0u;
                 candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
                 ++candidate_index)
            {
                uint32_t cache_slot_index;

                cache_slot_index = shared_cache_slots[candidate_index];
                if (cache_slot_index !=
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
                {
                    uint64_t cache_element_offset;

                    cache_element_offset =
                        (((uint64_t)cache_slot_index *
                          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                        (uint64_t)dimension_index;
                    accumulated_value +=
                        (shared_scores[candidate_index] / row_exponential_sum) *
                        SparkGlm52ResidentDecodeStageBf16ToFloat(
                            value_cache_bf16[cache_element_offset]);
                }
            }
        }
        output_value_bf16[
            output_row_offset + (uint64_t)dimension_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
    }
}


static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageAttentionOnlineKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint16_t *__restrict__ key_nope_cache_bf16,
    const uint16_t *__restrict__ value_cache_bf16,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint16_t *__restrict__ output_value_bf16,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_query[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION];
    __shared__ float shared_tile_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES];
    uint32_t sequence_index;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint64_t query_row_index;
    uint64_t sparse_row_offset;
    uint64_t output_row_offset;
    uint32_t dimension_index;
    uint32_t candidate_base;
    float online_maximum;
    float online_denominator;
    float accumulated_value;

    sequence_index = blockIdx.x;
    head_index = blockIdx.y;
    lane_index =
        threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index =
        threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    warp_count =
        blockDim.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    context_length = context_lengths[sequence_index];
    first_block_token_offset = first_block_token_offsets[sequence_index];
    query_row_index =
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
        (uint64_t)head_index;
    sparse_row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    output_row_offset =
        query_row_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;

    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[dimension_index] =
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                query_latent_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            dimension_index] = SparkGlm52ResidentDecodeStageBf16ToFloat(
                rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    __syncthreads();

    dimension_index = threadIdx.x;
    online_maximum = -FLT_MAX;
    online_denominator = 0.0f;
    accumulated_value = 0.0f;

    for (candidate_base = 0u;
         candidate_base < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         candidate_base += warp_count)
    {
        uint32_t candidate_index;
        uint32_t token_index;
        uint32_t cache_slot_index;
        float attention_score;
        uint32_t tile_index;

        candidate_index = candidate_base + warp_index;
        cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
        attention_score = -FLT_MAX;
        if (candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT)
        {
            token_index = sparse_token_indices[
                sparse_row_offset + (uint64_t)candidate_index];
            if (token_index < context_length)
            {
                cache_slot_index = SparkGlm52ResidentDecodeStageWarpResolveCacheSlot(
                    block_table,
                    sequence_index,
                    token_index,
                    first_block_token_offset,
                    block_token_count,
                    max_blocks_per_sequence,
                    kv_block_count,
                    cache_token_capacity,
                    lane_index);
            }
            if (cache_slot_index != SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                attention_score = SparkGlm52ResidentDecodeStageWarpDotProduct(
                    shared_query,
                    key_nope_cache_bf16,
                    mla_cache_bf16,
                    cache_slot_index,
                    head_index,
                    lane_index,
                    qk_scale);
            }
        }
        if (lane_index == 0u)
        {
            shared_tile_scores[warp_index] = attention_score;
            shared_tile_cache_slots[warp_index] = cache_slot_index;
        }
        __syncthreads();

        if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
        {
            for (tile_index = 0u; tile_index < warp_count; ++tile_index)
            {
                float tile_score;
                uint32_t tile_cache_slot;

                tile_score = shared_tile_scores[tile_index];
                tile_cache_slot = shared_tile_cache_slots[tile_index];
                if (tile_cache_slot !=
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT &&
                    tile_score > (-FLT_MAX * 0.5f))
                {
                    float next_maximum;
                    float old_scale;
                    float tile_scale;
                    float value;
                    uint64_t cache_element_offset;

                    next_maximum = tile_score > online_maximum
                        ? tile_score
                        : online_maximum;
                    old_scale = online_denominator > 0.0f
                        ? __expf(online_maximum - next_maximum)
                        : 0.0f;
                    tile_scale = __expf(tile_score - next_maximum);
                    cache_element_offset =
                        (((uint64_t)tile_cache_slot *
                          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                        (uint64_t)dimension_index;
                    value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                        value_cache_bf16[cache_element_offset]);
                    accumulated_value =
                        (accumulated_value * old_scale) + (value * tile_scale);
                    online_denominator =
                        (online_denominator * old_scale) + tile_scale;
                    online_maximum = next_maximum;
                }
            }
        }
        __syncthreads();
    }

    if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
    {
        output_value_bf16[output_row_offset + (uint64_t)dimension_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                online_denominator > 0.0f
                    ? accumulated_value / online_denominator
                    : 0.0f);
    }
}

static __global__ void SparkGlm52ResidentDecodeStageResidualKernel(
    const uint16_t *input_hidden_bf16,
    const uint16_t *projected_hidden_bf16,
    uint16_t *post_attention_hidden_bf16,
    uint32_t active_sequence_count)
{
    uint64_t element_index;
    uint64_t element_count;

    element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    element_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    while (element_index < element_count)
    {
        float input_value;
        float projected_value;

        input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_hidden_bf16[element_index]);
        projected_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            projected_hidden_bf16[element_index]);
        post_attention_hidden_bf16[element_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(input_value + projected_value);
        element_index += (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    }
}

static __device__ __forceinline__ bool SparkGlm52ResidentDecodeStageRouterCandidateIsBetter(
    float candidate_score,
    uint32_t candidate_expert,
    float current_score,
    uint32_t current_expert)
{
    return candidate_score > current_score ||
        (candidate_score == current_score && candidate_expert < current_expert);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeRouterTopKFromLogitsKernel(
    const float *__restrict__ router_logits_f32,
    const float *__restrict__ router_score_bias_f32,
    uint32_t *__restrict__ topk_expert_ids,
    float *__restrict__ topk_weights,
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    float routed_scaling_factor)
{
    __shared__ float shared_choice_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    __shared__ float shared_route_weights[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    __shared__ float shared_reduce_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    __shared__ float shared_reduce_weights[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    __shared__ uint32_t shared_reduce_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    __shared__ uint32_t shared_selected_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    __shared__ float shared_selected_weights[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    __shared__ float shared_selected_weight_sum;
    uint32_t sequence_index;
    uint32_t expert_index;
    uint32_t selected_index;

    sequence_index = blockIdx.x;
    expert_index = threadIdx.x;
    if (sequence_index >= active_sequence_count ||
        expert_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
    {
        return;
    }

    if (threadIdx.x == 0u)
    {
        shared_selected_weight_sum = 0.0f;
    }
    if (expert_index < expert_count)
    {
        float router_logit;
        float router_score;

        router_logit = router_logits_f32[
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT) +
            (uint64_t)expert_index];
        router_score = 1.0f / (1.0f + __expf(-router_logit));
        shared_route_weights[expert_index] = router_score;
        shared_choice_scores[expert_index] =
            router_score + router_score_bias_f32[expert_index];
    }
    else
    {
        shared_route_weights[expert_index] = 0.0f;
        shared_choice_scores[expert_index] = -FLT_MAX;
    }
    __syncthreads();

    if (top_k > SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K)
    {
        top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    }
    for (selected_index = 0u; selected_index < top_k; ++selected_index)
    {
        uint32_t reduce_stride;

        shared_reduce_scores[expert_index] = shared_choice_scores[expert_index];
        shared_reduce_weights[expert_index] = shared_route_weights[expert_index];
        shared_reduce_ids[expert_index] = expert_index;
        __syncthreads();

        for (reduce_stride = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT >> 1u;
             reduce_stride != 0u;
             reduce_stride >>= 1u)
        {
            if (expert_index < reduce_stride)
            {
                uint32_t other_index;
                float candidate_score;
                uint32_t candidate_expert;
                float current_score;
                uint32_t current_expert;

                other_index = expert_index + reduce_stride;
                candidate_score = shared_reduce_scores[other_index];
                candidate_expert = shared_reduce_ids[other_index];
                current_score = shared_reduce_scores[expert_index];
                current_expert = shared_reduce_ids[expert_index];
                if (SparkGlm52ResidentDecodeStageRouterCandidateIsBetter(
                        candidate_score,
                        candidate_expert,
                        current_score,
                        current_expert))
                {
                    shared_reduce_scores[expert_index] = candidate_score;
                    shared_reduce_ids[expert_index] = candidate_expert;
                    shared_reduce_weights[expert_index] =
                        shared_reduce_weights[other_index];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u)
        {
            uint32_t selected_expert;
            float selected_weight;

            selected_expert = shared_reduce_ids[0];
            selected_weight = shared_reduce_weights[0];
            shared_selected_ids[selected_index] = selected_expert;
            shared_selected_weights[selected_index] = selected_weight;
            shared_selected_weight_sum += selected_weight;
            if (selected_expert < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
            {
                shared_choice_scores[selected_expert] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    if (expert_index < top_k)
    {
        float selected_weight;
        uint64_t route_index;

        selected_weight = shared_selected_weights[expert_index];
        if (norm_topk_prob != 0u && shared_selected_weight_sum > 0.0f)
        {
            selected_weight /= shared_selected_weight_sum;
        }
        selected_weight *= routed_scaling_factor;
        route_index =
            ((uint64_t)sequence_index * (uint64_t)top_k) +
            (uint64_t)expert_index;
        topk_expert_ids[route_index] = shared_selected_ids[expert_index];
        topk_weights[route_index] = selected_weight;
    }
}


static __global__ void SparkGlm52ResidentDecodeStageSiluMulKernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    uint16_t *__restrict__ intermediate_bf16,
    uint64_t value_count)
{
    uint64_t value_index;
    uint64_t value_stride;

    value_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    value_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    while (value_index < value_count)
    {
        float gate_value;
        float up_value;
        float silu_value;

        gate_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            gate_bf16[value_index]);
        up_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            up_bf16[value_index]);
        silu_value = gate_value / (1.0f + __expf(-gate_value));
        intermediate_bf16[value_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(silu_value * up_value);
        value_index += value_stride;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageRestrictedLogitsKernel(
    const uint16_t *__restrict__ normalized_hidden_bf16,
    const uint16_t *__restrict__ restricted_lm_head_weight_bf16,
    float *__restrict__ restricted_logits,
    uint32_t active_sequence_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t token_row_index;
    uint32_t sequence_index;
    uint32_t hidden_index;
    float local_sum;
    float logit_sum;

    token_row_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (token_row_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT ||
        sequence_index >= active_sequence_count)
    {
        return;
    }
    local_sum = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float activation_value;
        float weight_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            normalized_hidden_bf16[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                (uint64_t)hidden_index]);
        weight_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            restricted_lm_head_weight_bf16[
                ((uint64_t)token_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                (uint64_t)hidden_index]);
        local_sum += activation_value * weight_value;
    }
    logit_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        restricted_logits[
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
            (uint64_t)token_row_index] = logit_sum;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageRestrictedArgmaxKernel(
    const float *__restrict__ restricted_logits,
    const uint32_t *__restrict__ restricted_token_ids,
    uint32_t *__restrict__ selected_token_ids,
    float *__restrict__ selected_token_scores,
    uint32_t active_sequence_count)
{
    __shared__ float shared_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    __shared__ uint32_t shared_tokens[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t sequence_index;
    uint32_t token_index;
    uint32_t stride;

    sequence_index = blockIdx.x;
    if (sequence_index >= active_sequence_count)
    {
        return;
    }
    token_index = threadIdx.x;
    if (token_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT)
    {
        shared_scores[token_index] = restricted_logits[
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
            (uint64_t)token_index];
        shared_tokens[token_index] = restricted_token_ids[token_index];
    }
    __syncthreads();
    for (stride =
             SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        if (token_index < stride)
        {
            float other_score;
            uint32_t other_token;

            other_score = shared_scores[token_index + stride];
            other_token = shared_tokens[token_index + stride];
            if (other_score > shared_scores[token_index] ||
                (other_score == shared_scores[token_index] &&
                 other_token < shared_tokens[token_index]))
            {
                shared_scores[token_index] = other_score;
                shared_tokens[token_index] = other_token;
            }
        }
        __syncthreads();
    }
    if (token_index == 0u)
    {
        selected_token_ids[sequence_index] = shared_tokens[0];
        selected_token_scores[sequence_index] = shared_scores[0];
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMtpDraftLogitsVectorizedKernel(
    const uint16_t *__restrict__ mtp_draft_hidden_bf16,
    const uint8_t *__restrict__ mtp_weight_payload_u8,
    const uint8_t *__restrict__ mtp_weight_scale_e8m0_u8,
    float *__restrict__ mtp_draft_logits,
    uint32_t active_sequence_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t restricted_index;
    uint32_t row_index;
    uint32_t sequence_index;
    uint32_t draft_index;
    uint32_t packed_hidden_index;
    uint64_t activation_row_offset;
    uint64_t packed_row_offset;
    uint64_t scale_row_offset;
    float local_sum;
    float logit_sum;

    restricted_index = blockIdx.x;
    row_index = blockIdx.y;
    sequence_index =
        row_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
    draft_index =
        row_index % SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
    if (restricted_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT ||
        sequence_index >= active_sequence_count)
    {
        return;
    }

    activation_row_offset =
        ((((uint64_t)sequence_index *
           (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
          (uint64_t)draft_index) *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
    packed_row_offset =
        (uint64_t)restricted_index *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
    scale_row_offset =
        (uint64_t)restricted_index *
        ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE);

    local_sum = 0.0f;
    for (packed_hidden_index = threadIdx.x;
         packed_hidden_index <
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
         packed_hidden_index += blockDim.x)
    {
        uint32_t hidden_index;
        uint8_t packed_weight;
        float scale_value;
        float first_activation;
        float second_activation;
        float first_weight;
        float second_weight;

        hidden_index = packed_hidden_index << 1u;
        packed_weight = mtp_weight_payload_u8[packed_row_offset +
            (uint64_t)packed_hidden_index];
        scale_value = SparkGlm52ResidentDecodeStageDecodeE8m0(
            mtp_weight_scale_e8m0_u8[
                scale_row_offset +
                ((uint64_t)hidden_index /
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE)]);
        first_activation = SparkGlm52ResidentDecodeStageBf16ToFloat(
            mtp_draft_hidden_bf16[activation_row_offset +
                (uint64_t)hidden_index]);
        second_activation = SparkGlm52ResidentDecodeStageBf16ToFloat(
            mtp_draft_hidden_bf16[activation_row_offset +
                (uint64_t)hidden_index + 1u]);
        first_weight = SparkGlm52ResidentDecodeStageDecodeE2m1(
            packed_weight & 0x0fu) * scale_value;
        second_weight = SparkGlm52ResidentDecodeStageDecodeE2m1(
            packed_weight >> 4u) * scale_value;
        local_sum += (first_activation * first_weight) +
            (second_activation * second_weight);
    }

    logit_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        mtp_draft_logits[
            ((uint64_t)row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
            (uint64_t)restricted_index] = logit_sum;
    }
}

static __global__ void SparkGlm52ResidentDecodeStageMtpArgmaxKernel(
    const float *__restrict__ mtp_draft_logits,
    const uint32_t *__restrict__ restricted_token_ids,
    uint32_t *__restrict__ mtp_draft_token_ids,
    uint32_t active_sequence_count)
{
    __shared__ float shared_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    __shared__ uint32_t shared_tokens[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t row_index;
    uint32_t token_index;
    uint32_t stride;
    uint32_t row_count;

    row_index = blockIdx.x;
    row_count =
        active_sequence_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
    if (row_index >= row_count)
    {
        return;
    }
    token_index = threadIdx.x;
    if (token_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT)
    {
        shared_scores[token_index] = mtp_draft_logits[
            ((uint64_t)row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
            (uint64_t)token_index];
        shared_tokens[token_index] = restricted_token_ids[token_index];
    }
    __syncthreads();
    for (stride =
             SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        if (token_index < stride)
        {
            float other_score;
            uint32_t other_token;

            other_score = shared_scores[token_index + stride];
            other_token = shared_tokens[token_index + stride];
            if (other_score > shared_scores[token_index] ||
                (other_score == shared_scores[token_index] &&
                 other_token < shared_tokens[token_index]))
            {
                shared_scores[token_index] = other_score;
                shared_tokens[token_index] = other_token;
            }
        }
        __syncthreads();
    }
    if (token_index == 0u)
    {
        mtp_draft_token_ids[row_index] = shared_tokens[0];
    }
}

static __global__ void SparkGlm52ResidentDecodeStageMtpVerifyCommitKernel(
    const uint32_t *__restrict__ target_token_ids,
    const uint32_t *__restrict__ draft_token_ids,
    uint32_t *__restrict__ accept_mask,
    uint32_t *__restrict__ committed_token_ids,
    uint32_t *__restrict__ event_counters,
    uint32_t active_sequence_count)
{
    uint32_t sequence_index;
    uint32_t accepting;
    uint32_t draft_index;

    sequence_index = blockIdx.x;
    if (threadIdx.x != 0u || sequence_index >= active_sequence_count)
    {
        return;
    }
    accepting = 1u;
    for (draft_index = 0u;
         draft_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
         ++draft_index)
    {
        uint32_t row_index;
        uint32_t accepted;

        row_index =
            (sequence_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
            draft_index;
        accepted = accepting != 0u &&
            draft_token_ids[row_index] == target_token_ids[row_index]
            ? 1u
            : 0u;
        accept_mask[row_index] = accepted;
        if (accepted != 0u)
        {
            committed_token_ids[row_index] = draft_token_ids[row_index];
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED],
                1u);
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                1u);
        }
        else if (accepting != 0u)
        {
            uint32_t rejected_suffix_count;
            uint32_t cancelled_suffix_count;

            rejected_suffix_count =
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT -
                draft_index;
            cancelled_suffix_count = rejected_suffix_count - 1u;
            committed_token_ids[row_index] = target_token_ids[row_index];
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED],
                rejected_suffix_count);
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                1u);
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK],
                rejected_suffix_count);
            atomicAdd(
                &event_counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED],
                cancelled_suffix_count);
            accepting = 0u;
        }
        else
        {
            committed_token_ids[row_index] =
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID;
        }
    }
}

static __global__ void SparkGlm52ResidentDecodeStageFusedFinalTokenTailKernel(
    const float *__restrict__ restricted_logits,
    const float *__restrict__ mtp_draft_logits,
    const uint32_t *__restrict__ restricted_token_ids,
    const uint32_t *__restrict__ mtp_target_token_ids,
    uint32_t *__restrict__ restricted_selected_token_ids,
    float *__restrict__ restricted_selected_token_scores,
    uint32_t *__restrict__ mtp_draft_token_ids,
    uint32_t *__restrict__ mtp_accept_mask,
    uint32_t *__restrict__ mtp_committed_token_ids,
    uint32_t *__restrict__ mtp_event_counters,
    uint32_t active_sequence_count)
{
    __shared__ float shared_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    __shared__ uint32_t shared_tokens[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t sequence_index;
    uint32_t token_index;
    uint32_t row_index;
    uint32_t stride;

    sequence_index = blockIdx.x;
    token_index = threadIdx.x;
    if (sequence_index >= active_sequence_count ||
        token_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT)
    {
        return;
    }

    shared_scores[0u][token_index] = restricted_logits[
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
        (uint64_t)token_index];
    shared_tokens[0u][token_index] = restricted_token_ids[token_index];
    for (row_index = 0u;
         row_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
         ++row_index)
    {
        uint64_t mtp_row_index;

        mtp_row_index =
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
            (uint64_t)row_index;
        shared_scores[row_index + 1u][token_index] = mtp_draft_logits[
            (mtp_row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
            (uint64_t)token_index];
        shared_tokens[row_index + 1u][token_index] = restricted_token_ids[token_index];
    }
    __syncthreads();

    for (stride = SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        if (token_index < stride)
        {
            for (row_index = 0u;
                 row_index <= SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
                 ++row_index)
            {
                float other_score;
                uint32_t other_token;

                other_score = shared_scores[row_index][token_index + stride];
                other_token = shared_tokens[row_index][token_index + stride];
                if (other_score > shared_scores[row_index][token_index] ||
                    (other_score == shared_scores[row_index][token_index] &&
                     other_token < shared_tokens[row_index][token_index]))
                {
                    shared_scores[row_index][token_index] = other_score;
                    shared_tokens[row_index][token_index] = other_token;
                }
            }
        }
        __syncthreads();
    }

    if (token_index == 0u)
    {
        uint32_t accepting;
        uint32_t draft_index;

        restricted_selected_token_ids[sequence_index] = shared_tokens[0u][0u];
        restricted_selected_token_scores[sequence_index] = shared_scores[0u][0u];
        accepting = 1u;
        for (draft_index = 0u;
             draft_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
             ++draft_index)
        {
            uint32_t mtp_row_index;
            uint32_t draft_token;
            uint32_t accepted;

            mtp_row_index =
                (sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
                draft_index;
            draft_token = shared_tokens[draft_index + 1u][0u];
            mtp_draft_token_ids[mtp_row_index] = draft_token;
            accepted = accepting != 0u &&
                draft_token == mtp_target_token_ids[mtp_row_index]
                ? 1u
                : 0u;
            mtp_accept_mask[mtp_row_index] = accepted;
            if (accepted != 0u)
            {
                mtp_committed_token_ids[mtp_row_index] = draft_token;
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED],
                    1u);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                    1u);
            }
            else if (accepting != 0u)
            {
                uint32_t rejected_suffix_count;
                uint32_t cancelled_suffix_count;

                rejected_suffix_count =
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT -
                    draft_index;
                cancelled_suffix_count = rejected_suffix_count - 1u;
                mtp_committed_token_ids[mtp_row_index] =
                    mtp_target_token_ids[mtp_row_index];
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED],
                    rejected_suffix_count);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                    1u);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK],
                    rejected_suffix_count);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED],
                    cancelled_suffix_count);
                accepting = 0u;
            }
            else
            {
                mtp_committed_token_ids[mtp_row_index] =
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID;
            }
        }
    }
}

static __device__ __forceinline__ bool SparkGlm52ResidentDecodeStageFinalCandidateIsBetter(
    float candidate_score,
    uint32_t candidate_token,
    float current_score,
    uint32_t current_token)
{
    return candidate_score > current_score ||
        (candidate_score == current_score && candidate_token < current_token);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageFusedFinalTokenCandidateKernel(
    const uint16_t *__restrict__ normalized_hidden_bf16,
    const uint16_t *__restrict__ restricted_lm_head_weight_bf16,
    const uint16_t *__restrict__ mtp_draft_hidden_bf16,
    const uint8_t *__restrict__ mtp_weight_payload_u8,
    const uint8_t *__restrict__ mtp_weight_scale_e8m0_u8,
    const uint32_t *__restrict__ restricted_token_ids,
    float *__restrict__ restricted_logits,
    float *__restrict__ candidate_scores,
    uint32_t *__restrict__ candidate_tokens,
    uint32_t active_sequence_count)
{
    __shared__ float partial_sums[SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE][8u];
    __shared__ float token_scores[SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE];
    __shared__ uint32_t token_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE];
    uint32_t sequence_index;
    uint32_t row_index;
    uint32_t group_index;
    uint32_t token_offset;
    uint32_t token_lane;
    uint32_t restricted_index;
    uint32_t hidden_index;
    uint64_t candidate_index;
    float local_sum;

    sequence_index = blockIdx.x;
    row_index = blockIdx.y;
    group_index = blockIdx.z;
    token_offset = threadIdx.x >> 3u;
    token_lane = threadIdx.x & 7u;
    restricted_index =
        (group_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE) +
        token_offset;
    if (sequence_index >= active_sequence_count ||
        row_index > SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
        group_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT ||
        token_offset >= SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE)
    {
        return;
    }

    local_sum = 0.0f;
    if (row_index == 0u)
    {
        for (hidden_index = token_lane;
             hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
             hidden_index += 8u)
        {
            float activation_value;
            float weight_value;

            activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                normalized_hidden_bf16[
                    ((uint64_t)sequence_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_index]);
            weight_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                restricted_lm_head_weight_bf16[
                    ((uint64_t)restricted_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_index]);
            local_sum += activation_value * weight_value;
        }
    }
    else
    {
        uint32_t draft_index;
        uint64_t activation_row_offset;
        uint64_t packed_row_offset;
        uint64_t scale_row_offset;
        uint32_t packed_hidden_index;

        draft_index = row_index - 1u;
        activation_row_offset =
            ((((uint64_t)sequence_index *
               (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
              (uint64_t)draft_index) *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
        packed_row_offset =
            (uint64_t)restricted_index *
            ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
        scale_row_offset =
            (uint64_t)restricted_index *
            ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE);
        for (packed_hidden_index = token_lane;
             packed_hidden_index <
                 (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
             packed_hidden_index += 8u)
        {
            uint32_t first_hidden_index;
            uint8_t packed_weight;
            float scale_value;
            float first_activation;
            float second_activation;
            float first_weight;
            float second_weight;

            first_hidden_index = packed_hidden_index << 1u;
            packed_weight = mtp_weight_payload_u8[packed_row_offset +
                (uint64_t)packed_hidden_index];
            scale_value = SparkGlm52ResidentDecodeStageDecodeE8m0(
                mtp_weight_scale_e8m0_u8[
                    scale_row_offset +
                    ((uint64_t)first_hidden_index /
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE)]);
            first_activation = SparkGlm52ResidentDecodeStageBf16ToFloat(
                mtp_draft_hidden_bf16[activation_row_offset +
                    (uint64_t)first_hidden_index]);
            second_activation = SparkGlm52ResidentDecodeStageBf16ToFloat(
                mtp_draft_hidden_bf16[activation_row_offset +
                    (uint64_t)first_hidden_index + 1u]);
            first_weight = SparkGlm52ResidentDecodeStageDecodeE2m1(
                packed_weight & 0x0fu) * scale_value;
            second_weight = SparkGlm52ResidentDecodeStageDecodeE2m1(
                packed_weight >> 4u) * scale_value;
            local_sum += (first_activation * first_weight) +
                (second_activation * second_weight);
        }
    }

    partial_sums[token_offset][token_lane] = local_sum;
    __syncthreads();
    if (token_lane == 0u)
    {
        float token_score;

        token_score = partial_sums[token_offset][0u] +
            partial_sums[token_offset][1u] +
            partial_sums[token_offset][2u] +
            partial_sums[token_offset][3u] +
            partial_sums[token_offset][4u] +
            partial_sums[token_offset][5u] +
            partial_sums[token_offset][6u] +
            partial_sums[token_offset][7u];
        token_scores[token_offset] = token_score;
        token_ids[token_offset] = restricted_token_ids[restricted_index];
        if (row_index == 0u && restricted_logits != 0)
        {
            restricted_logits[
                ((uint64_t)sequence_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT) +
                (uint64_t)restricted_index] = token_score;
        }
    }
    __syncthreads();
    {
        uint32_t stride;

        for (stride =
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_SIZE >> 1u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                float other_score;
                uint32_t other_token;

                other_score = token_scores[threadIdx.x + stride];
                other_token = token_ids[threadIdx.x + stride];
                if (SparkGlm52ResidentDecodeStageFinalCandidateIsBetter(
                        other_score,
                        other_token,
                        token_scores[threadIdx.x],
                        token_ids[threadIdx.x]))
                {
                    token_scores[threadIdx.x] = other_score;
                    token_ids[threadIdx.x] = other_token;
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0u)
    {
        candidate_index =
            ((((uint64_t)sequence_index *
               (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u)) +
              (uint64_t)row_index) *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT) +
            (uint64_t)group_index;
        candidate_scores[candidate_index] = token_scores[0u];
        candidate_tokens[candidate_index] = token_ids[0u];
    }
}

static __global__ void SparkGlm52ResidentDecodeStageFusedFinalTokenCommitKernel(
    const float *__restrict__ candidate_scores,
    const uint32_t *__restrict__ candidate_tokens,
    const uint32_t *__restrict__ mtp_target_token_ids,
    uint32_t *__restrict__ restricted_selected_token_ids,
    float *__restrict__ restricted_selected_token_scores,
    uint32_t *__restrict__ mtp_draft_token_ids,
    uint32_t *__restrict__ mtp_accept_mask,
    uint32_t *__restrict__ mtp_committed_token_ids,
    uint32_t *__restrict__ mtp_event_counters,
    uint32_t active_sequence_count)
{
    __shared__ float shared_scores[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT];
    __shared__ uint32_t shared_tokens[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT];
    uint32_t sequence_index;
    uint32_t row_index;
    uint32_t group_index;
    uint32_t stride;

    sequence_index = blockIdx.x;
    group_index = threadIdx.x;
    if (sequence_index >= active_sequence_count ||
        group_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT)
    {
        return;
    }
    for (row_index = 0u;
         row_index <= SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
         ++row_index)
    {
        uint64_t candidate_index;

        candidate_index =
            ((((uint64_t)sequence_index *
               (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u)) +
              (uint64_t)row_index) *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT) +
            (uint64_t)group_index;
        shared_scores[row_index][group_index] = candidate_scores[candidate_index];
        shared_tokens[row_index][group_index] = candidate_tokens[candidate_index];
    }
    __syncthreads();

    for (stride = SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT >> 1u;
         stride != 0u;
         stride >>= 1u)
    {
        if (group_index < stride)
        {
            for (row_index = 0u;
                 row_index <= SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
                 ++row_index)
            {
                float other_score;
                uint32_t other_token;

                other_score = shared_scores[row_index][group_index + stride];
                other_token = shared_tokens[row_index][group_index + stride];
                if (SparkGlm52ResidentDecodeStageFinalCandidateIsBetter(
                        other_score,
                        other_token,
                        shared_scores[row_index][group_index],
                        shared_tokens[row_index][group_index]))
                {
                    shared_scores[row_index][group_index] = other_score;
                    shared_tokens[row_index][group_index] = other_token;
                }
            }
        }
        __syncthreads();
    }

    if (group_index == 0u)
    {
        uint32_t accepting;
        uint32_t draft_index;

        restricted_selected_token_ids[sequence_index] = shared_tokens[0u][0u];
        restricted_selected_token_scores[sequence_index] = shared_scores[0u][0u];
        accepting = 1u;
        for (draft_index = 0u;
             draft_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
             ++draft_index)
        {
            uint32_t mtp_row_index;
            uint32_t draft_token;
            uint32_t accepted;

            mtp_row_index =
                (sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
                draft_index;
            draft_token = shared_tokens[draft_index + 1u][0u];
            mtp_draft_token_ids[mtp_row_index] = draft_token;
            accepted = accepting != 0u &&
                draft_token == mtp_target_token_ids[mtp_row_index]
                ? 1u
                : 0u;
            mtp_accept_mask[mtp_row_index] = accepted;
            if (accepted != 0u)
            {
                mtp_committed_token_ids[mtp_row_index] = draft_token;
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED],
                    1u);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                    1u);
            }
            else if (accepting != 0u)
            {
                uint32_t rejected_suffix_count;
                uint32_t cancelled_suffix_count;

                rejected_suffix_count =
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT -
                    draft_index;
                cancelled_suffix_count = rejected_suffix_count - 1u;
                mtp_committed_token_ids[mtp_row_index] =
                    mtp_target_token_ids[mtp_row_index];
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED],
                    rejected_suffix_count);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED],
                    1u);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK],
                    rejected_suffix_count);
                atomicAdd(
                    &mtp_event_counters[
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED],
                    cancelled_suffix_count);
                accepting = 0u;
            }
            else
            {
                mtp_committed_token_ids[mtp_row_index] =
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID;
            }
        }
    }
}

static uint32_t SparkGlm52ResidentDecodeStagePrepareBlockCount(
    uint32_t active_sequence_count)
{
    uint64_t rope_pair_count;
    uint64_t total_work_count;
    uint64_t block_count;

    rope_pair_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
    total_work_count =
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
         rope_pair_count) +
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
        ((uint64_t)active_sequence_count * rope_pair_count) +
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION);
    block_count =
        (total_work_count +
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS;
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return (uint32_t)block_count;
}

static uint32_t SparkGlm52ResidentDecodeStageElementBlockCount(
    uint64_t element_count)
{
    uint64_t block_count;

    block_count =
        (element_count +
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS;
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return (uint32_t)block_count;
}

static SparkStatus SparkGlm52ResidentDecodeStageCheckCudaLaunch(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream)
{
    cudaError_t cuda_status;

    if (node_context == 0 ||
        node_context->launch_check_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE)
    {
        return SPARK_STATUS_OK;
    }

    cuda_status = node_context->launch_check_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_PEEK
        ? cudaPeekAtLastError()
        : cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        if (getenv("GLM52_CUDA_LAUNCH_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "spark_glm52_cuda_launch_error mode=%u error=%d message=%s\n",
                node_context->launch_check_mode,
                (int)cuda_status,
                cudaGetErrorString(cuda_status));
        }
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->launch_error_count += 1u;
        }
        if (node_context->launch_check_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR)
        {
            cudaStreamSynchronize(cuda_stream);
        }
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageMaybeMarkPhase(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t phase_index)
{
    if (node_context->phase_clock_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 ||
        pipeline_slot->phase_clock_cycles == 0)
    {
        return SPARK_STATUS_OK;
    }
    SparkGlm52ResidentDecodeStageMarkPhaseKernel<<<1u, 1u, 0u, cuda_stream>>>(
        pipeline_slot->phase_clock_cycles,
        phase_index);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}


typedef SparkStatus (*SparkGlm52ResidentDecodeStageCustomLinearLaunchFunction)(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageRestrictedLogitsLaunchFunction)(
    const SparkGlm52ResidentDecodeStageRestrictedLogitsPlan *restricted_logits_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageMtpDraftLaunchFunction)(
    const SparkGlm52ResidentDecodeStageMtpDraftPlan *mtp_draft_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageFullStageLaunchFunction)(
    const SparkGlm52ResidentDecodeStageFullStagePlan *full_stage_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageFp8MoeLaunchFunction)(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageStageSliceLaunchFunction)(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageExactStageSliceLaunchFunction)(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageExactFinalTokenTailLaunchFunction)(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

typedef SparkStatus (*SparkGlm52ResidentDecodeStageExactStageMoeLaunchFunction)(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

static bool SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t active_sequence_count)
{
    if (linear_plan == 0)
    {
        return false;
    }
    return linear_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION &&
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED &&
        linear_plan->input_dimension == input_dimension &&
        linear_plan->output_dimension == output_dimension &&
        active_sequence_count <= linear_plan->maximum_active_sequence_count;
}

static bool SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
    uint32_t plan_kind)
{
    switch (plan_kind)
    {
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR:
    case SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR:
        return true;
    default:
        return false;
    }
}


static uint64_t SparkGlm52ResidentDecodeStageDivideRoundUpU64(
    uint64_t value,
    uint64_t divisor)
{
    if (divisor == 0u)
    {
        return 0u;
    }
    return (value + divisor - 1u) / divisor;
}

static uint32_t SparkGlm52ResidentDecodeStageReferencePlanWeightFormat(
    uint32_t plan_kind)
{
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_FP8_E4M3_ROW_MAJOR ||
        plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    }
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_NVFP4_E2M1_ROW_MAJOR ||
        plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1;
    }
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_MXFP4_E2M1_ROW_MAJOR ||
        plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16;
}

static uint32_t SparkGlm52ResidentDecodeStageReferencePlanScaleBlockSize(
    uint32_t weight_format)
{
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    }
    if (weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    }
    return 0u;
}

static bool SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52ResidentDecodeStageQuantizedLinearView **view_out)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *view;
    uint64_t weight_element_count;
    uint64_t input_scale_block_count;
    uint64_t output_scale_block_count;
    uint64_t scale_element_count;
    uint64_t required_payload_bytes;
    uint64_t required_scale_bytes;
    uint32_t expected_weight_format;
    uint32_t expected_scale_block_size;

    if (view_out != 0)
    {
        *view_out = 0;
    }
    if (linear_plan == 0 || linear_plan->custom_state == 0)
    {
        return false;
    }

    expected_weight_format = SparkGlm52ResidentDecodeStageReferencePlanWeightFormat(
        linear_plan->plan_kind);
    expected_scale_block_size = SparkGlm52ResidentDecodeStageReferencePlanScaleBlockSize(
        expected_weight_format);
    if (expected_scale_block_size == 0u)
    {
        return false;
    }

    view = (const SparkGlm52ResidentDecodeStageQuantizedLinearView *)
        linear_plan->custom_state;
    if (view->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QUANTIZED_LINEAR_VIEW_ABI_VERSION ||
        view->weight_format != expected_weight_format ||
        view->input_dimension != linear_plan->input_dimension ||
        view->output_dimension != linear_plan->output_dimension ||
        view->scale_block_size != expected_scale_block_size ||
        view->output_is_f32 != linear_plan->output_is_f32 ||
        view->weight_payload == 0 ||
        view->weight_scale == 0)
    {
        return false;
    }

    weight_element_count =
        (uint64_t)view->input_dimension * (uint64_t)view->output_dimension;
    if (view->weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        required_payload_bytes = weight_element_count;
    }
    else
    {
        required_payload_bytes = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
            weight_element_count,
            2u);
    }

    input_scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
        view->input_dimension,
        view->scale_block_size);
    output_scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU64(
        view->output_dimension,
        view->scale_block_size);
    scale_element_count = input_scale_block_count * output_scale_block_count;
    required_scale_bytes =
        view->weight_format ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3
            ? scale_element_count * (uint64_t)sizeof(float)
            : scale_element_count;

    if (view->weight_payload_bytes < required_payload_bytes ||
        view->weight_scale_bytes < required_scale_bytes)
    {
        return false;
    }
    if (view_out != 0)
    {
        *view_out = view;
    }
    return true;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchQuantizedReferenceLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    void *output,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *view;
    cudaError_t cuda_status;
    dim3 grid;

    if (input == 0 || output == 0 ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &view))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (active_sequence_count == 0u ||
        active_sequence_count > linear_plan->maximum_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    grid = dim3(
        linear_plan->output_dimension,
        active_sequence_count,
        1u);
    if (linear_plan->output_is_f32 != 0u)
    {
        SparkGlm52ResidentDecodeStageQuantizedLinearF32Kernel<<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)input,
            (const uint8_t *)view->weight_payload,
            view->weight_scale,
            (float *)output,
            active_sequence_count,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            view->weight_format,
            view->scale_block_size);
    }
    else
    {
        SparkGlm52ResidentDecodeStageQuantizedLinearBf16Kernel<<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)input,
            (const uint8_t *)view->weight_payload,
            view->weight_scale,
            (uint16_t *)output,
            active_sequence_count,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            view->weight_format,
            view->scale_block_size);
    }

    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess &&
        getenv("GLM52_LINEAR_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "quantized_tensor_core_linear_launch_failed kind=%u in=%u out=%u active=%u grid=(%u,%u,%u) error=%d message=%s\n",
            linear_plan->plan_kind,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            active_sequence_count,
            grid.x,
            grid.y,
            grid.z,
            (int)cuda_status,
            cudaGetErrorString(cuda_status));
    }
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}


static bool SparkGlm52ResidentDecodeStageBlackwellQuantizedTensorCorePlanShapeIsSupported(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    if (linear_plan == 0)
    {
        return false;
    }
    return linear_plan->input_dimension != 0u &&
        linear_plan->output_dimension != 0u &&
        (linear_plan->input_dimension & 15u) == 0u &&
        (linear_plan->output_dimension & 15u) == 0u &&
        linear_plan->maximum_active_sequence_count != 0u &&
        linear_plan->maximum_active_sequence_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
}

static bool SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
    uint32_t plan_kind)
{
    return plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR ||
        plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR ||
        plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    void *output,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;
    dim3 grid;
    cudaError_t cuda_status;
    uint32_t sequence_tile_rows;

    if (linear_plan == 0 || input == 0 || output == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > linear_plan->maximum_active_sequence_count ||
        !SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            linear_plan->plan_kind) ||
        !SparkGlm52ResidentDecodeStageBlackwellQuantizedTensorCorePlanShapeIsSupported(
            linear_plan) ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &quantized_view))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    sequence_tile_rows = SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M;
    if (linear_plan->input_dimension ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION &&
        linear_plan->output_dimension ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
    {
        if (active_sequence_count > 32u)
        {
            sequence_tile_rows = 64u;
        }
        else if (active_sequence_count > 16u)
        {
            sequence_tile_rows = 32u;
        }
    }
    grid = dim3(
        (linear_plan->output_dimension +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N - 1u) /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
        (active_sequence_count + sequence_tile_rows - 1u) / sequence_tile_rows,
        1u);
    if (sequence_tile_rows == 64u)
    {
        SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearWideKernel<64u, 16u><<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)input,
            (const uint8_t *)quantized_view->weight_payload,
            quantized_view->weight_scale,
            output,
            active_sequence_count,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            quantized_view->weight_format,
            quantized_view->scale_block_size,
            linear_plan->output_is_f32);
    }
    else if (sequence_tile_rows == 32u)
    {
        SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearWideKernel<32u, 16u><<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)input,
            (const uint8_t *)quantized_view->weight_payload,
            quantized_view->weight_scale,
            output,
            active_sequence_count,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            quantized_view->weight_format,
            quantized_view->scale_block_size,
            linear_plan->output_is_f32);
    }
    else
    {
        SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearKernel<<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)input,
            (const uint8_t *)quantized_view->weight_payload,
            quantized_view->weight_scale,
            output,
            active_sequence_count,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            quantized_view->weight_format,
            quantized_view->scale_block_size,
            linear_plan->output_is_f32);
    }

    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    void *cuda_stream)
{
    (void)weight;
    if (cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52ResidentDecodeStageLaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan(
        linear_plan,
        input,
        output,
        active_sequence_count,
        (cudaStream_t)cuda_stream);
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateBlackwellNativeQuantizedTensorCoreWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    if (!SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            linear_plan == 0 ? 0u : linear_plan->plan_kind) ||
        !SparkGlm52ResidentDecodeStageBlackwellQuantizedTensorCorePlanShapeIsSupported(
            linear_plan))
    {
        return 0u;
    }
    return 0u;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedTensorCoreLinearPlan(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;

    if (linear_plan == 0 ||
        linear_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION ||
        !SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            linear_plan->plan_kind) ||
        !SparkGlm52ResidentDecodeStageBlackwellQuantizedTensorCorePlanShapeIsSupported(
            linear_plan) ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &quantized_view))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    (void)quantized_view;
    linear_plan->custom_launch_function =
        (void *)SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Sm121RequiredDecodeStageLinearPlanIndexIsRawProjection(
    uint32_t plan_index)
{
    return plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedProjectionPlans(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plans,
    uint32_t linear_plan_count)
{
    uint32_t plan_index;
    uint32_t bound_plan_count;

    if (linear_plans == 0 ||
        linear_plan_count < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    bound_plan_count = 0u;
    for (plan_index = 0u;
         plan_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;
         ++plan_index)
    {
        SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;

        if (!SparkGlm52Sm121RequiredDecodeStageLinearPlanIndexIsRawProjection(
                plan_index))
        {
            continue;
        }
        linear_plan = &linear_plans[plan_index];
        if (linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR ||
            linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR ||
            linear_plan->plan_kind ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
        {
            SparkStatus status;

            status = SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedTensorCoreLinearPlan(
                linear_plan);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            bound_plan_count += 1u;
        }
    }
    return bound_plan_count == 0u
        ? SPARK_STATUS_NOT_FOUND
        : SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream);



static const SparkGlm52ResidentDecodeStageB12xMoePlan *SparkGlm52ResidentDecodeStageGetB12xMoePlan(
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan)
{
    if (b12x_moe_dispatch_plan == 0 ||
        b12x_moe_dispatch_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X ||
        b12x_moe_dispatch_plan->opaque_state == 0)
    {
        return 0;
    }
    return (const SparkGlm52ResidentDecodeStageB12xMoePlan *)
        b12x_moe_dispatch_plan->opaque_state;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateB12xMoePlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan,
    const SparkGlm52ResidentDecodeStageB12xMoePlan **b12x_plan_out)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    uint32_t required_capabilities;
    uint64_t required_route_count;

    if (b12x_plan_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *b12x_plan_out = 0;
    if (node_context == 0 || b12x_moe_dispatch_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    b12x_plan = SparkGlm52ResidentDecodeStageGetB12xMoePlan(b12x_moe_dispatch_plan);
    if (b12x_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_REQUIRED_CAPABILITIES;
    required_route_count =
        (uint64_t)node_context->max_active_sequence_count *
        (uint64_t)node_context->moe_top_k;

    if (required_route_count > UINT32_MAX ||
        b12x_moe_dispatch_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_ABI_VERSION ||
        b12x_moe_dispatch_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X ||
        b12x_moe_dispatch_plan->reserved != 0u ||
        b12x_moe_dispatch_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_moe_dispatch_plan->maximum_route_count < required_route_count ||
        b12x_moe_dispatch_plan->expert_count !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT ||
        b12x_moe_dispatch_plan->top_k !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K ||
        b12x_moe_dispatch_plan->intermediate_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION ||
        b12x_moe_dispatch_plan->validated_maximum_latency_ns == 0u ||
        b12x_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION ||
        b12x_plan->reserved0 != 0u ||
        b12x_plan->reserved1 != 0u ||
        b12x_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        b12x_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->expert_count !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT ||
        b12x_plan->top_k !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K ||
        b12x_plan->hidden_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION ||
        b12x_plan->intermediate_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION ||
        b12x_plan->gate_up_order !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE ||
        b12x_plan->weight_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW ||
        b12x_plan->scale_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE ||
        b12x_plan->quant_mode !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4 ||
        b12x_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        b12x_plan->cuda_architecture != 121u ||
        b12x_plan->validated_maximum_latency_ns == 0u ||
        b12x_plan->state_cell == 0 ||
        b12x_plan->w1_weight_fp4_static_view == 0 ||
        b12x_plan->w1_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w1_alpha_fp32_by_expert == 0 ||
        b12x_plan->fc2_input_scale_fp32_by_expert == 0 ||
        b12x_plan->w2_weight_fp4_static_view == 0 ||
        b12x_plan->w2_scale_static_storage_ue4m3 == 0 ||
        b12x_plan->w2_alpha_fp32_by_expert == 0 ||
        (b12x_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (b12x_plan->recipe.abi_version !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION ||
        b12x_plan->recipe.hidden_dimension != b12x_plan->hidden_dimension ||
        b12x_plan->recipe.intermediate_dimension !=
            b12x_plan->intermediate_dimension ||
        b12x_plan->recipe.expert_count != b12x_plan->expert_count ||
        b12x_plan->recipe.top_k != b12x_plan->top_k ||
        b12x_plan->recipe.maximum_token_count <
            node_context->max_active_sequence_count ||
        b12x_plan->recipe.gate_up_order != b12x_plan->gate_up_order ||
        b12x_plan->recipe.weight_layout != b12x_plan->weight_layout ||
        b12x_plan->recipe.scale_layout != b12x_plan->scale_layout ||
        b12x_plan->recipe.quant_mode != b12x_plan->quant_mode ||
        b12x_plan->recipe.output_dtype != b12x_plan->output_dtype ||
        b12x_plan->recipe.cuda_architecture != b12x_plan->cuda_architecture ||
        b12x_plan->recipe.qualified_maximum_microseconds == 0u ||
        b12x_plan->recipe.qualification_record_hash_low64 == 0u ||
        b12x_plan->recipe.kernel_manifest_hash_low64 == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *b12x_plan_out = b12x_plan;
    return SPARK_STATUS_OK;
}


static SparkStatus SparkGlm52ResidentDecodeStageLaunchMoeRouterLogitsForB12x(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *router_plan;

    if (node_context == 0 || pipeline_slot == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        node_context->linear_plans == 0 ||
        node_context->linear_plan_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS ||
        node_context->moe_router_weight_bf16 == 0 ||
        pipeline_slot->post_attention_normalized_hidden_bf16 == 0 ||
        pipeline_slot->moe_router_logits == 0 ||
        node_context->moe_expert_count == 0u ||
        node_context->moe_expert_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    router_plan = &node_context->linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS];
    if (!SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            router_plan,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->moe_expert_count,
            active_sequence_count) ||
        router_plan->output_is_f32 == 0u ||
        !SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
            router_plan->plan_kind))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
        router_plan,
        pipeline_slot->post_attention_normalized_hidden_bf16,
        node_context->moe_router_weight_bf16,
        pipeline_slot->moe_router_logits,
        active_sequence_count,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchMoeRouterForB12x(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkStatus status;

    if (node_context == 0 || pipeline_slot == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        node_context->moe_router_score_bias_f32 == 0 ||
        node_context->moe_routed_scaling_factor == 0.0f ||
        pipeline_slot->moe_topk_expert_ids == 0 ||
        pipeline_slot->moe_topk_weights == 0 ||
        node_context->moe_expert_count == 0u ||
        node_context->moe_expert_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        node_context->moe_top_k == 0u ||
        node_context->moe_top_k > node_context->moe_expert_count ||
        node_context->moe_top_k > SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageLaunchMoeRouterLogitsForB12x(
        node_context,
        pipeline_slot,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ResidentDecodeStageMoeRouterTopKFromLogitsKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        pipeline_slot->moe_router_logits,
        node_context->moe_router_score_bias_f32,
        pipeline_slot->moe_topk_expert_ids,
        pipeline_slot->moe_topk_weights,
        active_sequence_count,
        node_context->moe_expert_count,
        node_context->moe_top_k,
        node_context->moe_norm_topk_prob,
        node_context->moe_routed_scaling_factor);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchValidatedFlashInferB12xMoe(
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan,
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream)
{
    SparkGlm52Sm121FlashInferB12xMoeArguments arguments;
    uint64_t hidden_element_count;
    SparkStatus status;

    if (b12x_moe_dispatch_plan == 0 || b12x_plan == 0 ||
        node_context == 0 || pipeline_slot == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u || b12x_plan->state_cell == 0 ||
        *(b12x_plan->state_cell) == 0 ||
        node_context->moe_router_score_bias_f32 == 0 ||
        node_context->moe_routed_scaling_factor == 0.0f ||
        pipeline_slot->moe_router_logits == 0 ||
        pipeline_slot->moe_topk_expert_ids == 0 ||
        pipeline_slot->moe_topk_weights == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (active_sequence_count > b12x_plan->maximum_token_count ||
        active_sequence_count > b12x_moe_dispatch_plan->maximum_active_sequence_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkGlm52ResidentDecodeStageLaunchMoeRouterLogitsForB12x(
        node_context,
        pipeline_slot,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&arguments, 0, sizeof(arguments));
    arguments.abi_version =
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION;
    arguments.token_count = active_sequence_count;
    arguments.maximum_token_count = b12x_plan->maximum_token_count;
    arguments.expert_count = b12x_plan->expert_count;
    arguments.top_k = b12x_plan->top_k;
    arguments.hidden_dimension = b12x_plan->hidden_dimension;
    arguments.intermediate_dimension = b12x_plan->intermediate_dimension;
    arguments.argument_flags =
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS |
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_DETERMINISTIC_FC2_FINALIZE;
    arguments.hidden_bf16 = pipeline_slot->post_attention_normalized_hidden_bf16;
    arguments.topk_ids_i32 = (int32_t *)pipeline_slot->moe_topk_expert_ids;
    arguments.topk_weights_fp32 = pipeline_slot->moe_topk_weights;
    arguments.router_logits_f32 = pipeline_slot->moe_router_logits;
    arguments.router_score_bias_f32 = node_context->moe_router_score_bias_f32;
    arguments.router_norm_topk_prob = node_context->moe_norm_topk_prob;
    arguments.router_routed_scaling_factor =
        node_context->moe_routed_scaling_factor;
    arguments.w1_weight_fp4_static_view = b12x_plan->w1_weight_fp4_static_view;
    arguments.w1_scale_static_storage_ue4m3 =
        b12x_plan->w1_scale_static_storage_ue4m3;
    arguments.w1_alpha_fp32_by_expert = b12x_plan->w1_alpha_fp32_by_expert;
    arguments.fc2_input_scale_fp32_by_expert =
        b12x_plan->fc2_input_scale_fp32_by_expert;
    arguments.w2_weight_fp4_static_view = b12x_plan->w2_weight_fp4_static_view;
    arguments.w2_scale_static_storage_ue4m3 =
        b12x_plan->w2_scale_static_storage_ue4m3;
    arguments.w2_alpha_fp32_by_expert = b12x_plan->w2_alpha_fp32_by_expert;
    arguments.output_bf16 = pipeline_slot->moe_route_output_bf16;
    arguments.workspace = b12x_plan->workspace;
    arguments.workspace_bytes = b12x_plan->workspace_bytes;
    arguments.cuda_stream = (void *)cuda_stream;

    status = SparkGlm52Sm121FlashInferB12xMoeLaunch(
        *(b12x_plan->state_cell),
        &arguments);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    hidden_element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    SparkGlm52ResidentDecodeStageResidualKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)pipeline_slot->moe_route_output_bf16,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count);
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52ResidentDecodeStageLaunchFlashInferB12xMoe(
    const SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan *b12x_moe_dispatch_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    SparkStatus status;

    if (cuda_stream_pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ResidentDecodeStageValidateB12xMoePlan(
        node_context,
        b12x_moe_dispatch_plan,
        &b12x_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageLaunchValidatedFlashInferB12xMoe(
        b12x_moe_dispatch_plan,
        b12x_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        (cudaStream_t)cuda_stream_pointer);
}




static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MoePlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    uint32_t required_capabilities;

    if (node_context == 0 || fp8_moe_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES;
    if (fp8_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION ||
        fp8_moe_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->maximum_token_count <
            node_context->max_active_sequence_count ||
        fp8_moe_plan->expert_count != node_context->moe_expert_count ||
        fp8_moe_plan->top_k != node_context->moe_top_k ||
        fp8_moe_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        fp8_moe_plan->intermediate_dimension !=
            node_context->moe_intermediate_dimension ||
        fp8_moe_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        fp8_moe_plan->cuda_architecture != 121u ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->launch_function == 0 ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        fp8_moe_plan->validated_maximum_latency_ns == 0u ||
        (fp8_moe_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52ResidentDecodeStageLaunchFp8Moe(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    SparkGlm52ResidentDecodeStageFp8MoeLaunchFunction launch_function;
    cudaStream_t cuda_stream;
    uint64_t hidden_element_count;
    SparkStatus status;

    if (pipeline_slot == 0 || cuda_stream_pointer == 0 ||
        active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ResidentDecodeStageValidateFp8MoePlan(
        node_context,
        fp8_moe_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (active_sequence_count > fp8_moe_plan->maximum_token_count ||
        active_sequence_count > fp8_moe_plan->maximum_active_sequence_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    cuda_stream = (cudaStream_t)cuda_stream_pointer;
    status = SparkGlm52ResidentDecodeStageLaunchMoeRouterLogitsForB12x(
        node_context,
        pipeline_slot,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    launch_function = (SparkGlm52ResidentDecodeStageFp8MoeLaunchFunction)
        fp8_moe_plan->launch_function;
    status = launch_function(
        fp8_moe_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    hidden_element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    SparkGlm52ResidentDecodeStageResidualKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)pipeline_slot->moe_route_output_bf16,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRequiredFp8Moe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkStatus status;

    if (node_context == 0 || pipeline_slot == 0 ||
        node_context->fp8_moe_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ResidentDecodeStageLaunchFp8Moe(
        node_context->fp8_moe_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
}

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchFullStagePlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    bool *plan_was_launched)
{
    const SparkGlm52ResidentDecodeStageFullStagePlan *full_stage_plan;
    SparkGlm52ResidentDecodeStageFullStageLaunchFunction launch_function;
    SparkStatus status;

    if (plan_was_launched == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *plan_was_launched = false;
    if (node_context == 0 || node_context->full_stage_plan == 0)
    {
        if ((node_context != 0) &&
            ((node_context->reserved_execution_flags &
              SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN) != 0u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }

    full_stage_plan = node_context->full_stage_plan;
    if (full_stage_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_PLAN_ABI_VERSION ||
        full_stage_plan->reserved != 0u ||
        full_stage_plan->launch_function == 0 ||
        active_sequence_count > full_stage_plan->maximum_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    launch_function = (SparkGlm52ResidentDecodeStageFullStageLaunchFunction)
        full_stage_plan->launch_function;
    status = launch_function(
        full_stage_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "stage_slice_plan_launch_returned status=%d launched=1\n",
            (int)status);
    }
    *plan_was_launched = true;
    if (cuda_slot_state != 0)
    {
        cuda_slot_state->launch_chain_count += 1u;
        if ((full_stage_plan->capability_flags &
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_CUDA_GRAPH_REPLAY) != 0u)
        {
            cuda_slot_state->graph_replay_count += 1u;
        }
    }
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}


static void SparkGlm52ResidentDecodeStageBuildRuntimeKvLayerContexts(
    const SparkGlm52ResidentDecodeStageNodeContext *const *source_layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    SparkGlm52ResidentDecodeStageNodeContext *runtime_node_contexts,
    SparkGlm52ResidentDecodeStagePipelineSlot *runtime_pipeline_slots,
    const SparkGlm52ResidentDecodeStageNodeContext **runtime_layer_node_contexts)
{
    const SparkGlm52ResidentDecodeStageNodeContext *source_node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot *slot_base;
    uint32_t layer_index;
    uint32_t slot_index;

    if (source_layer_node_contexts == 0 ||
        runtime_kv_block_table == 0 ||
        runtime_node_contexts == 0 ||
        runtime_pipeline_slots == 0 ||
        runtime_layer_node_contexts == 0)
    {
        return;
    }

    for (layer_index = 0u; layer_index < layer_count; ++layer_index)
    {
        source_node_context = source_layer_node_contexts[layer_index];
        runtime_node_contexts[layer_index] = *source_node_context;
        slot_base = runtime_pipeline_slots +
            ((uint64_t)layer_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
        for (slot_index = 0u;
             slot_index < source_node_context->pipeline_slot_count;
             ++slot_index)
        {
            slot_base[slot_index] = source_node_context->pipeline_slots[slot_index];
        }
        slot_base[pipeline_slot_index].block_table =
            runtime_kv_block_table->physical_block_indices;
        runtime_node_contexts[layer_index].pipeline_slots = slot_base;
        runtime_layer_node_contexts[layer_index] = &runtime_node_contexts[layer_index];
    }
}

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchStageSlicePlan(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    bool *plan_was_launched)
{
    SparkGlm52ResidentDecodeStageStageSliceLaunchFunction launch_function;
    SparkGlm52ResidentDecodeStageNodeContext runtime_node_contexts[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
    const SparkGlm52ResidentDecodeStageNodeContext *runtime_layer_node_contexts[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT];
    const SparkGlm52ResidentDecodeStageNodeContext *const *effective_layer_node_contexts;
    uint32_t required_capabilities;
    uint32_t has_builtin_exact_pp13;
    SparkStatus status;

    if (plan_was_launched == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *plan_was_launched = false;
    if (stage_slice_plan == 0)
    {
        return SPARK_STATUS_OK;
    }

    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_REQUIRED_CAPABILITIES;
    has_builtin_exact_pp13 =
        (stage_slice_plan->capability_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) != 0u
        ? 1u
        : 0u;
    if (stage_slice_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION ||
        stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
        stage_slice_plan->maximum_layer_count < layer_count ||
        stage_slice_plan->maximum_layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        (stage_slice_plan->launch_function == 0 &&
            has_builtin_exact_pp13 == 0u) ||
        stage_slice_plan->validated_maximum_latency_ns == 0u ||
        (stage_slice_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    effective_layer_node_contexts = layer_node_contexts;
    if (runtime_kv_block_table != 0)
    {
        SparkGlm52ResidentDecodeStageBuildRuntimeKvLayerContexts(
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            runtime_kv_block_table,
            runtime_node_contexts,
            runtime_pipeline_slots,
            runtime_layer_node_contexts);
        effective_layer_node_contexts = runtime_layer_node_contexts;
    }

    if (stage_slice_plan->launch_function != 0)
    {
        launch_function =
            (SparkGlm52ResidentDecodeStageStageSliceLaunchFunction)
                stage_slice_plan->launch_function;
        status = launch_function(
            stage_slice_plan,
            effective_layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            active_sequence_count,
            final_token_stage,
            (void *)cuda_stream);
    }
    else
    {
        status = SparkGlm52Sm121RequiredDecodeStageLaunchStageSlice(
            stage_slice_plan,
            effective_layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            active_sequence_count,
            final_token_stage,
            runtime_kv_block_table,
            0,
            (void *)cuda_stream);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *plan_was_launched = true;
    if (cuda_slot_state != 0 &&
        (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_INTERNAL_CUDA_GRAPH_COUNTERS) == 0u)
    {
        cuda_slot_state->launch_chain_count += 1u;
        if ((stage_slice_plan->capability_flags &
             SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_CUDA_GRAPH_REPLAY) != 0u)
        {
            cuda_slot_state->graph_replay_count += 1u;
        }
    }
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        layer_node_contexts[0],
        cuda_slot_state,
        cuda_stream);
    if (getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "stage_slice_plan_post_check status=%d\n",
            (int)status);
    }
    return status;
}

static const SparkGlm52ResidentDecodeStageLinearPlan *SparkGlm52ResidentDecodeStageGetLinearPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;

    if (node_context->linear_plans == 0 ||
        plan_index >= node_context->linear_plan_count)
    {
        return 0;
    }
    linear_plan = &node_context->linear_plans[plan_index];
    if (linear_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION ||
        linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED ||
        linear_plan->input_dimension != input_dimension ||
        linear_plan->output_dimension != output_dimension ||
        active_sequence_count > linear_plan->maximum_active_sequence_count)
    {
        return 0;
    }
    return linear_plan;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream)
{
    if (linear_plan == 0 || input == 0 || output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (linear_plan->plan_kind ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM)
    {
        SparkGlm52ResidentDecodeStageCustomLinearLaunchFunction launch_function;

        launch_function =
            (SparkGlm52ResidentDecodeStageCustomLinearLaunchFunction)
                linear_plan->custom_launch_function;
        if (launch_function == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return launch_function(
            linear_plan,
            input,
            weight,
            output,
            active_sequence_count,
            (void *)cuda_stream);
    }
    if (SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            linear_plan->plan_kind))
    {
        SparkGlm52ResidentDecodeStageCustomLinearLaunchFunction launch_function;

        launch_function =
            (SparkGlm52ResidentDecodeStageCustomLinearLaunchFunction)
                linear_plan->custom_launch_function;
        if (launch_function != 0)
        {
            return launch_function(
                linear_plan,
                input,
                weight,
                output,
                active_sequence_count,
                (void *)cuda_stream);
        }
        return SparkGlm52ResidentDecodeStageLaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan(
            linear_plan,
            input,
            output,
            active_sequence_count,
            cuda_stream);
    }
    if (linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_FP8_E4M3_ROW_MAJOR ||
        linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_NVFP4_E2M1_ROW_MAJOR ||
        linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUDA_REFERENCE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SparkGlm52ResidentDecodeStageLaunchQuantizedReferenceLinearPlan(
            linear_plan,
            input,
            output,
            active_sequence_count,
            cuda_stream);
    }

#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
    if (linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR ||
        linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR)
    {
        cublasStatus_t cublas_status;
        float alpha;
        float beta;

        if (weight == 0 ||
            linear_plan->cublaslt_handle == 0 ||
            linear_plan->matmul_descriptor == 0 ||
            linear_plan->input_layout == 0 ||
            linear_plan->weight_layout == 0 ||
            linear_plan->output_layout == 0 ||
            linear_plan->algorithm == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        alpha = linear_plan->alpha != 0.0f ? linear_plan->alpha : 1.0f;
        beta = linear_plan->beta;
        cublas_status = cublasLtMatmul(
            (cublasLtHandle_t)linear_plan->cublaslt_handle,
            (cublasLtMatmulDesc_t)linear_plan->matmul_descriptor,
            &alpha,
            input,
            (cublasLtMatrixLayout_t)linear_plan->input_layout,
            weight,
            (cublasLtMatrixLayout_t)linear_plan->weight_layout,
            &beta,
            output,
            (cublasLtMatrixLayout_t)linear_plan->output_layout,
            output,
            (cublasLtMatrixLayout_t)linear_plan->output_layout,
            (const cublasLtMatmulAlgo_t *)linear_plan->algorithm,
            linear_plan->workspace,
            (size_t)linear_plan->workspace_bytes,
            cuda_stream);
        return cublas_status == CUBLAS_STATUS_SUCCESS
            ? SPARK_STATUS_OK
            : SPARK_STATUS_INTERNAL_ERROR;
    }
#endif
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    cudaStream_t cuda_stream,
    bool plan_is_required,
    bool *plan_was_launched)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;
    SparkStatus status;

    if (plan_was_launched != 0)
    {
        *plan_was_launched = false;
    }
    linear_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
        node_context,
        plan_index,
        input_dimension,
        output_dimension,
        active_sequence_count);
    if (linear_plan == 0)
    {
        if (plan_is_required && getenv("GLM52_LINEAR_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "linear_plan_missing index=%u in=%u out=%u active=%u count=%u\n",
                plan_index,
                input_dimension,
                output_dimension,
                active_sequence_count,
                node_context != 0 ? node_context->linear_plan_count : 0u);
        }
        return plan_is_required ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }
    if (active_sequence_count != linear_plan->maximum_active_sequence_count)
    {
        if (plan_is_required && getenv("GLM52_LINEAR_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "linear_plan_active_mismatch index=%u in=%u out=%u active=%u max=%u kind=%u\n",
                plan_index,
                input_dimension,
                output_dimension,
                active_sequence_count,
                linear_plan->maximum_active_sequence_count,
                linear_plan->plan_kind);
        }
        return plan_is_required ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }
    if (plan_was_launched != 0)
    {
        *plan_was_launched = true;
    }
    status = SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
        linear_plan,
        input,
        weight,
        output,
        active_sequence_count,
        cuda_stream);
    if (status != SPARK_STATUS_OK &&
        getenv("GLM52_LINEAR_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "linear_plan_launch_failed index=%u kind=%u in=%u out=%u active=%u max=%u output_f32=%u status=%d\n",
            plan_index,
            linear_plan->plan_kind,
            linear_plan->input_dimension,
            linear_plan->output_dimension,
            active_sequence_count,
            linear_plan->maximum_active_sequence_count,
            linear_plan->output_is_f32,
            (int)status);
    }
    return status;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchLinear(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    const uint16_t *input_bf16,
    const uint16_t *weight_bf16,
    uint16_t *output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t plan_index,
    bool plan_is_required)
{
    dim3 grid;
    bool plan_was_launched;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(
        node_context,
        plan_index,
        input_bf16,
        weight_bf16,
        output_bf16,
        active_sequence_count,
        input_dimension,
        output_dimension,
        cuda_stream,
        plan_is_required,
        &plan_was_launched);
    if (status != SPARK_STATUS_OK || plan_was_launched)
    {
        return status;
    }

    grid = dim3(output_dimension, active_sequence_count, 1u);
    SparkGlm52ResidentDecodeStageBf16LinearKernel<<<
        grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        input_bf16,
        weight_bf16,
        output_bf16,
        active_sequence_count,
        input_dimension,
        output_dimension);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchLinearFp8(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    const uint16_t *input_bf16,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    uint16_t *output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t plan_index,
    bool plan_is_required)
{
    dim3 grid;
    bool plan_was_launched;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(
        node_context,
        plan_index,
        input_bf16,
        weight_fp8_e4m3,
        output_bf16,
        active_sequence_count,
        input_dimension,
        output_dimension,
        cuda_stream,
        plan_is_required,
        &plan_was_launched);
    if (status != SPARK_STATUS_OK || plan_was_launched)
    {
        return status;
    }

    grid = dim3(output_dimension, active_sequence_count, 1u);
    SparkGlm52ResidentDecodeStageFp8LinearKernel<<<
        grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        input_bf16,
        weight_fp8_e4m3,
        weight_scale_inv_f32,
        output_bf16,
        active_sequence_count,
        input_dimension,
        output_dimension);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRawLinear(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    const uint16_t *input_bf16,
    const uint16_t *weight_bf16,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    uint16_t *output_bf16,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t plan_index)
{
    bool plan_is_required;

    plan_is_required = node_context->projection_backend_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_SCALAR_REFERENCE;
    if (node_context->projection_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3)
    {
        return SparkGlm52ResidentDecodeStageLaunchLinearFp8(
            node_context,
            cuda_slot_state,
            cuda_stream,
            input_bf16,
            weight_fp8_e4m3,
            weight_scale_inv_f32,
            output_bf16,
            active_sequence_count,
            input_dimension,
            output_dimension,
            plan_index,
            plan_is_required);
    }
    if (node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_NVFP4_E2M1 ||
        node_context->projection_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_MXFP4_E2M1)
    {
        return SparkGlm52ResidentDecodeStageLaunchLinear(
            node_context,
            cuda_slot_state,
            cuda_stream,
            input_bf16,
            0,
            output_bf16,
            active_sequence_count,
            input_dimension,
            output_dimension,
            plan_index,
            true);
    }
    return SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        input_bf16,
        weight_bf16,
        output_bf16,
        active_sequence_count,
        input_dimension,
        output_dimension,
        plan_index,
        plan_is_required);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRmsNormDimension(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    const uint16_t *input_bf16,
    const uint16_t *weight_bf16,
    uint16_t *output_bf16,
    uint32_t active_sequence_count,
    uint32_t dimension)
{
    SparkGlm52ResidentDecodeStageRmsNormDimensionKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        input_bf16,
        weight_bf16,
        output_bf16,
        active_sequence_count,
        dimension,
        node_context->rms_norm_epsilon);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageTraceProjectionStatus(
    const char *phase_name,
    SparkStatus status)
{
    if (status != SPARK_STATUS_OK &&
        getenv("GLM52_PROJECTION_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "projection_failed phase=%s status=%d\n",
            phase_name != 0 ? phase_name : "unknown",
            (int)status);
    }
    return status;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRawGlmProjection(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    uint64_t raw_query_map_count;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->raw_query_a_weight_bf16,
        node_context->raw_query_a_weight_fp8_e4m3,
        node_context->raw_query_a_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_query_a_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_query_a",
            status);
    }
    status = SparkGlm52ResidentDecodeStageLaunchRmsNormDimension(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->raw_query_a_bf16,
        (const uint16_t *)node_context->raw_query_a_norm_weight_bf16,
        (uint16_t *)pipeline_slot->raw_query_a_normalized_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_query_a_norm",
            status);
    }
    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->raw_query_a_normalized_bf16,
        (const uint16_t *)node_context->raw_query_b_weight_bf16,
        node_context->raw_query_b_weight_fp8_e4m3,
        node_context->raw_query_b_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_query_b_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_query_b",
            status);
    }
    raw_query_map_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    SparkGlm52ResidentDecodeStageMapRawQueryKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(raw_query_map_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->raw_query_b_bf16,
        (uint16_t *)pipeline_slot->query_latent_bf16,
        (uint16_t *)pipeline_slot->query_rope_input_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_query_map",
            status);
    }
    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->raw_kv_a_weight_bf16,
        node_context->raw_kv_a_weight_fp8_e4m3,
        node_context->raw_kv_a_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_kv_a_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_kv_a",
            status);
    }
    SparkGlm52ResidentDecodeStageSplitRawKvAKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(
            (uint64_t)active_sequence_count *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->raw_kv_a_bf16,
        (uint16_t *)pipeline_slot->current_kv_latent_bf16,
        (uint16_t *)pipeline_slot->key_rope_input_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_kv_a_split",
            status);
    }
    status = SparkGlm52ResidentDecodeStageLaunchRmsNormDimension(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->current_kv_latent_bf16,
        (const uint16_t *)node_context->raw_kv_a_norm_weight_bf16,
        (uint16_t *)pipeline_slot->raw_kv_a_normalized_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
            "raw_kv_a_norm",
            status);
    }
    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->raw_kv_a_normalized_bf16,
        (const uint16_t *)node_context->raw_kv_b_weight_bf16,
        node_context->raw_kv_b_weight_fp8_e4m3,
        node_context->raw_kv_b_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_kv_b_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B);
    return SparkGlm52ResidentDecodeStageTraceProjectionStatus(
        "raw_kv_b",
        status);
}

static uint32_t SparkGlm52ResidentDecodeStageExactStageSliceCanOverlapRawProjection(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (exact_stage_slice_plan == 0 || node_context == 0)
    {
        return 0u;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_QKV_BRANCH_OVERLAP) == 0u)
    {
        return 0u;
    }
    if (exact_stage_slice_plan->query_branch_stream == 0 ||
        exact_stage_slice_plan->kv_branch_stream == 0 ||
        exact_stage_slice_plan->branch_ready_event == 0 ||
        exact_stage_slice_plan->query_branch_event == 0 ||
        exact_stage_slice_plan->kv_branch_event == 0)
    {
        return 0u;
    }
    return node_context->projection_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRawGlmProjectionWithBranchOverlap(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    cudaStream_t query_stream;
    cudaStream_t kv_stream;
    cudaEvent_t branch_ready_event;
    cudaEvent_t query_done_event;
    cudaEvent_t kv_done_event;
    uint64_t raw_query_map_count;
    SparkStatus status;
    cudaError_t cuda_status;

    if (!SparkGlm52ResidentDecodeStageExactStageSliceCanOverlapRawProjection(
            exact_stage_slice_plan,
            node_context))
    {
        return SparkGlm52ResidentDecodeStageLaunchRawGlmProjection(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }

    query_stream = (cudaStream_t)exact_stage_slice_plan->query_branch_stream;
    kv_stream = (cudaStream_t)exact_stage_slice_plan->kv_branch_stream;
    branch_ready_event = (cudaEvent_t)exact_stage_slice_plan->branch_ready_event;
    query_done_event = (cudaEvent_t)exact_stage_slice_plan->query_branch_event;
    kv_done_event = (cudaEvent_t)exact_stage_slice_plan->kv_branch_event;

    cuda_status = cudaEventRecord(branch_ready_event, cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaStreamWaitEvent(query_stream, branch_ready_event, 0u);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaStreamWaitEvent(kv_stream, branch_ready_event, 0u);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        query_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->raw_query_a_weight_bf16,
        node_context->raw_query_a_weight_fp8_e4m3,
        node_context->raw_query_a_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_query_a_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchRmsNormDimension(
        node_context,
        cuda_slot_state,
        query_stream,
        (const uint16_t *)pipeline_slot->raw_query_a_bf16,
        (const uint16_t *)node_context->raw_query_a_norm_weight_bf16,
        (uint16_t *)pipeline_slot->raw_query_a_normalized_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        query_stream,
        (const uint16_t *)pipeline_slot->raw_query_a_normalized_bf16,
        (const uint16_t *)node_context->raw_query_b_weight_bf16,
        node_context->raw_query_b_weight_fp8_e4m3,
        node_context->raw_query_b_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_query_b_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    raw_query_map_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    SparkGlm52ResidentDecodeStageMapRawQueryKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(raw_query_map_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        query_stream>>>(
        (const uint16_t *)pipeline_slot->raw_query_b_bf16,
        (uint16_t *)pipeline_slot->query_latent_bf16,
        (uint16_t *)pipeline_slot->query_rope_input_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        query_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        kv_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->raw_kv_a_weight_bf16,
        node_context->raw_kv_a_weight_fp8_e4m3,
        node_context->raw_kv_a_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_kv_a_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlm52ResidentDecodeStageSplitRawKvAKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(
            (uint64_t)active_sequence_count *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        kv_stream>>>(
        (const uint16_t *)pipeline_slot->raw_kv_a_bf16,
        (uint16_t *)pipeline_slot->current_kv_latent_bf16,
        (uint16_t *)pipeline_slot->key_rope_input_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        kv_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchRmsNormDimension(
        node_context,
        cuda_slot_state,
        kv_stream,
        (const uint16_t *)pipeline_slot->current_kv_latent_bf16,
        (const uint16_t *)node_context->raw_kv_a_norm_weight_bf16,
        (uint16_t *)pipeline_slot->raw_kv_a_normalized_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        kv_stream,
        (const uint16_t *)pipeline_slot->raw_kv_a_normalized_bf16,
        (const uint16_t *)node_context->raw_kv_b_weight_bf16,
        node_context->raw_kv_b_weight_fp8_e4m3,
        node_context->raw_kv_b_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->raw_kv_b_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_status = cudaEventRecord(query_done_event, query_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaEventRecord(kv_done_event, kv_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaStreamWaitEvent(cuda_stream, query_done_event, 0u);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaStreamWaitEvent(cuda_stream, kv_done_event, 0u);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchLoweredProjection(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->query_latent_weight_bf16,
        (uint16_t *)pipeline_slot->query_latent_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_LATENT,
        node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->query_rope_weight_bf16,
        (uint16_t *)pipeline_slot->query_rope_input_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_ROPE,
        node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->key_rope_weight_bf16,
        (uint16_t *)pipeline_slot->key_rope_input_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KEY_ROPE,
        node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->kv_latent_weight_bf16,
        (uint16_t *)pipeline_slot->current_kv_latent_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KV_LATENT,
        node_context->projection_backend_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    if (node_context->sparse_index_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED)
    {
        return SPARK_STATUS_OK;
    }
    if (node_context->sparse_index_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX)
    {
        SparkGlm52ResidentDecodeStageCopyContextPrefixSparseIndicesKernel<<<
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            pipeline_slot->context_lengths,
            pipeline_slot->sparse_token_indices,
            active_sequence_count);
        return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
    }

    SparkGlm52ResidentDecodeStageDsaSelectKernel<<<
        active_sequence_count,
        1u,
        0u,
        cuda_stream>>>(
        pipeline_slot->dsa_token_scores,
        pipeline_slot->context_lengths,
        pipeline_slot->sparse_token_indices,
        active_sequence_count,
        node_context->dsa_candidate_count);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}


static SparkStatus SparkGlm52ResidentDecodeStageLaunchPreboundDenseMlp(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    uint64_t intermediate_value_count;
    uint64_t hidden_element_count;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
        (const uint16_t *)node_context->dense_gate_weight_bf16,
        (uint16_t *)pipeline_slot->moe_gate_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
        true);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
        (const uint16_t *)node_context->dense_up_weight_bf16,
        (uint16_t *)pipeline_slot->moe_up_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
        true);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    intermediate_value_count =
        (uint64_t)active_sequence_count *
        (uint64_t)node_context->dense_intermediate_dimension;
    SparkGlm52ResidentDecodeStageSiluMulKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(intermediate_value_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->moe_gate_bf16,
        (const uint16_t *)pipeline_slot->moe_up_bf16,
        (uint16_t *)pipeline_slot->moe_intermediate_bf16,
        intermediate_value_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->moe_intermediate_bf16,
        (const uint16_t *)node_context->dense_down_weight_bf16,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
        true);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    hidden_element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    SparkGlm52ResidentDecodeStageResidualKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRequiredB12xMoe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkStatus status;

    if (node_context == 0 || pipeline_slot == 0 ||
        node_context->b12x_moe_dispatch_plan == 0 ||
        node_context->b12x_moe_dispatch_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_ABI_VERSION ||
        node_context->b12x_moe_dispatch_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageLaunchFlashInferB12xMoe(
        node_context->b12x_moe_dispatch_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (cuda_slot_state != 0)
    {
        cuda_slot_state->b12x_moe_success_count += 1u;
    }
    return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
}



static uint32_t SparkGlm52ResidentDecodeStageExactPlanUsesBuiltinFusedStageMoe(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    if (exact_stage_slice_plan == 0)
    {
        return 0u;
    }
    return (exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
        (exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u &&
        exact_stage_slice_plan->fused_moe_launch_function == 0;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchExternalExactStageMoeLayer(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    SparkGlm52ResidentDecodeStageExactStageMoeLaunchFunction launch_function;
    SparkStatus status;

    if (exact_stage_slice_plan == 0 ||
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) == 0u ||
        exact_stage_slice_plan->fused_moe_launch_function == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (node_context == 0 || pipeline_slot == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    launch_function = (SparkGlm52ResidentDecodeStageExactStageMoeLaunchFunction)
        exact_stage_slice_plan->fused_moe_launch_function;
    status = launch_function(
        exact_stage_slice_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchBuiltinFusedStageMoeLayer(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    SparkStatus status;

    if (!SparkGlm52ResidentDecodeStageExactPlanUsesBuiltinFusedStageMoe(
            exact_stage_slice_plan))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (node_context == 0 || pipeline_slot == 0 || active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        if (node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE ||
            node_context->b12x_moe_dispatch_plan == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkGlm52ResidentDecodeStageValidateB12xMoePlan(
            node_context,
            node_context->b12x_moe_dispatch_plan,
            &b12x_plan);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52ResidentDecodeStageLaunchValidatedFlashInferB12xMoe(
            node_context->b12x_moe_dispatch_plan,
            b12x_plan,
            node_context,
            pipeline_slot,
            active_sequence_count,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->b12x_moe_success_count += 1u;
        }
        return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        if (node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE ||
            node_context->fp8_moe_plan == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkGlm52ResidentDecodeStageLaunchFp8Moe(
            node_context->fp8_moe_plan,
            node_context,
            pipeline_slot,
            active_sequence_count,
            (void *)cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchPostAttentionMlp(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    SparkStatus status;

    if (node_context == 0 || pipeline_slot == 0 || active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY)
    {
        return SPARK_STATUS_OK;
    }

    SparkGlm52ResidentDecodeStageRmsNormKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)node_context->post_attention_norm_weight_bf16,
        (uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
        active_sequence_count,
        node_context->rms_norm_epsilon);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_POST_ATTENTION_NORM);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchExternalExactStageMoeLayer(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        exact_stage_slice_plan);
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchBuiltinFusedStageMoeLayer(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        exact_stage_slice_plan);
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        if (node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE &&
            node_context->mlp_execution_mode !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkGlm52ResidentDecodeStageLaunchPreboundDenseMlp(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        status = SparkGlm52ResidentDecodeStageLaunchMoeRouterForB12x(
            node_context,
            pipeline_slot,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkGlm52ResidentDecodeStageMaybeMarkPhase(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE);
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        return SparkGlm52ResidentDecodeStageLaunchRequiredB12xMoe(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }

    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        return SparkGlm52ResidentDecodeStageLaunchRequiredFp8Moe(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}



static SparkStatus SparkGlm52ResidentDecodeStageLaunchMtpDraft(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkGlm52ResidentDecodeStageMtpDraftLaunchFunction launch_function;
    dim3 mtp_logits_grid;
    SparkStatus status;

    if (node_context->mtp_draft_plan != 0 &&
        node_context->mtp_draft_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_PLAN_ABI_VERSION &&
        node_context->mtp_draft_plan->launch_function != 0)
    {
        launch_function =
            (SparkGlm52ResidentDecodeStageMtpDraftLaunchFunction)
                node_context->mtp_draft_plan->launch_function;
        status = launch_function(
            node_context->mtp_draft_plan,
            node_context,
            pipeline_slot,
            active_sequence_count,
            (void *)cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
    }
    if ((node_context->reserved_execution_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MTP_DRAFT) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    mtp_logits_grid = dim3(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        active_sequence_count *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,
        1u);
    SparkGlm52ResidentDecodeStageMtpDraftLogitsVectorizedKernel<<<
        mtp_logits_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->mtp_draft_hidden_bf16,
        node_context->mtp_mxfp4_weight_payload_u8,
        node_context->mtp_mxfp4_scale_e8m0_u8,
        pipeline_slot->mtp_draft_logits,
        active_sequence_count);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchRestrictedLogits(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkGlm52ResidentDecodeStageRestrictedLogitsLaunchFunction launch_function;
    bool plan_was_launched;
    SparkStatus status;
    dim3 restricted_grid;

    status = SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS,
        pipeline_slot->normalized_hidden_bf16,
        node_context->restricted_lm_head_weight_bf16,
        pipeline_slot->restricted_logits,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        cuda_stream,
        (node_context->reserved_execution_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS) != 0u,
        &plan_was_launched);
    if (status != SPARK_STATUS_OK || plan_was_launched)
    {
        return status;
    }

    if (node_context->restricted_logits_plan != 0 &&
        node_context->restricted_logits_plan->abi_version ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_LOGITS_PLAN_ABI_VERSION &&
        node_context->restricted_logits_plan->launch_function != 0)
    {
        launch_function =
            (SparkGlm52ResidentDecodeStageRestrictedLogitsLaunchFunction)
                node_context->restricted_logits_plan->launch_function;
        status = launch_function(
            node_context->restricted_logits_plan,
            node_context,
            pipeline_slot,
            active_sequence_count,
            (void *)cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
    }
    if ((node_context->reserved_execution_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    restricted_grid = dim3(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        active_sequence_count,
        1u);
    SparkGlm52ResidentDecodeStageRestrictedLogitsKernel<<<
        restricted_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->restricted_lm_head_weight_bf16,
        pipeline_slot->restricted_logits,
        active_sequence_count);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static void CUDART_CB SparkGlm52ResidentDecodeStageCudaCompletion(
    void *completion_context)
{
    SparkGlm52ResidentDecodeStageBackendCompletion *completion;

    completion =
        (SparkGlm52ResidentDecodeStageBackendCompletion *)completion_context;
    if (completion != 0 && completion->function != 0)
    {
        completion->function(completion->context);
    }
}

static SparkStatus SparkGlm52ResidentDecodeStageEnqueueCompletion(
    cudaStream_t cuda_stream,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    cudaError_t cuda_status;

    if (completion == 0)
    {
        return SPARK_STATUS_OK;
    }
    if (completion->function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_status = cudaLaunchHostFunc(
        cuda_stream,
        SparkGlm52ResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->launch_error_count += 1u;
        }
        cudaStreamSynchronize(cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static uint64_t SparkGlm52ResidentDecodeStageMixGraphSignature(
    uint64_t signature,
    uint64_t value)
{
    signature ^= value;
    signature *= 1099511628211ull;
    return signature;
}

static uint64_t SparkGlm52ResidentDecodeStagePointerGraphSignature(
    const void *pointer)
{
    return (uint64_t)(uintptr_t)pointer;
}

static uint64_t SparkGlm52ResidentDecodeStageMixLinearPlansGraphSignature(
    uint64_t signature,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan;
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;
    uint32_t plan_index;
    uint32_t plan_count;

    if (node_context == 0 || node_context->linear_plans == 0)
    {
        return signature;
    }

    plan_count = node_context->linear_plan_count;
    if (plan_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT)
    {
        plan_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT;
    }
    for (plan_index = 0u; plan_index < plan_count; ++plan_index)
    {
        linear_plan = &node_context->linear_plans[plan_index];
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->abi_version);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->plan_kind);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->input_dimension);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->output_dimension);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->maximum_active_sequence_count);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->output_is_f32);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->cublaslt_handle));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->matmul_descriptor));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->algorithm));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->workspace));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            linear_plan->workspace_bytes);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->custom_launch_function));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                linear_plan->custom_state));

        if (SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
                linear_plan,
                &quantized_view))
        {
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                quantized_view->weight_format);
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                quantized_view->scale_block_size);
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                SparkGlm52ResidentDecodeStagePointerGraphSignature(
                    quantized_view->weight_payload));
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                SparkGlm52ResidentDecodeStagePointerGraphSignature(
                    quantized_view->weight_scale));
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                quantized_view->weight_payload_bytes);
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                quantized_view->weight_scale_bytes);
        }
    }
    return signature;
}

static uint32_t SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context->kv_block_token_count != 0u)
    {
        return node_context->kv_block_token_count;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
}

static uint64_t SparkGlm52ResidentDecodeStageComputeGraphSignature(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t hidden_output_only)
{
    uint64_t signature;

    signature = 1469598103934665603ull;
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, hidden_output_only);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->projection_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->layer_progression_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->sparse_index_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->projection_backend_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->mlp_execution_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->attention_execution_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->reserved_execution_flags);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->model_quantization_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->linear_plan_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->linear_plans));
    signature = SparkGlm52ResidentDecodeStageMixLinearPlansGraphSignature(signature, node_context);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->b12x_moe_dispatch_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->restricted_logits_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->mtp_draft_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->full_stage_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->bulk_prefill_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->fp8_moe_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->mla_cache_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->key_nope_cache_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->value_cache_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->attention_norm_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_query_a_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_query_a_weight_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_query_a_norm_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_query_b_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_query_b_weight_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_kv_a_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_kv_a_weight_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_kv_a_norm_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_kv_b_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->raw_kv_b_weight_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->attention_output_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->attention_output_weight_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->post_attention_norm_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->moe_router_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->moe_router_score_bias_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->final_norm_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->restricted_lm_head_weight_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->mtp_mxfp4_weight_payload_u8));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->mtp_mxfp4_scale_e8m0_u8));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->input_hidden_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->layer_output_hidden_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->post_attention_hidden_bf16));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->block_table));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->context_lengths));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->first_block_token_offsets));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(pipeline_slot->restricted_logits));
    return signature;
}

static SparkStatus SparkGlm52ResidentDecodeStageFinishSubmit(
    cudaStream_t cuda_stream,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    uint32_t graph_capture_active,
    uint64_t graph_specialization_signature,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    cudaGraph_t captured_graph;
    cudaGraphExec_t captured_graph_exec;
    cudaError_t cuda_status;

    captured_graph = 0;
    captured_graph_exec = 0;
    if (graph_capture_active != 0u)
    {
        cuda_status = cudaStreamEndCapture(cuda_stream, &captured_graph);
        if (cuda_status != cudaSuccess)
        {
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_status = cudaGraphInstantiate(
            &captured_graph_exec,
            captured_graph,
            0,
            0,
            0);
        cudaGraphDestroy(captured_graph);
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->cuda_graph_exec = (void *)captured_graph_exec;
        cuda_slot_state->graph_specialization_signature =
            graph_specialization_signature;
        cuda_status = cudaGraphLaunch(captured_graph_exec, cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->graph_replay_count += 1u;
    }
    return SparkGlm52ResidentDecodeStageEnqueueCompletion(
        cuda_stream,
        cuda_slot_state,
        completion);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchFusedFinalTokenTail(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkGlm52ResidentDecodeStageExactFinalTokenTailLaunchFunction
        final_token_launch_function;
    SparkStatus status;

    if (exact_stage_slice_plan != 0 &&
        exact_stage_slice_plan->final_token_launch_function != 0)
    {
        final_token_launch_function =
            (SparkGlm52ResidentDecodeStageExactFinalTokenTailLaunchFunction)
                exact_stage_slice_plan->final_token_launch_function;
        return final_token_launch_function(
            exact_stage_slice_plan,
            node_context,
            pipeline_slot,
            active_sequence_count,
            (void *)cuda_stream);
    }

    SparkGlm52ResidentDecodeStageFusedFinalTokenTailKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        0u,
        cuda_stream>>>(
        pipeline_slot->restricted_logits,
        pipeline_slot->mtp_draft_logits,
        node_context->restricted_token_ids,
        pipeline_slot->mtp_target_token_ids,
        pipeline_slot->restricted_selected_token_ids,
        pipeline_slot->restricted_selected_token_scores,
        pipeline_slot->mtp_draft_token_ids,
        pipeline_slot->mtp_accept_mask,
        pipeline_slot->mtp_committed_token_ids,
        pipeline_slot->mtp_event_counters,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    return status;
}

static bool SparkGlm52ResidentDecodeStageExactPlanUsesBuiltInFusedFinalTokenEpilogue(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    uint64_t candidate_count;
    uint64_t required_workspace_bytes;

    if (exact_stage_slice_plan == 0 ||
        exact_stage_slice_plan->final_token_launch_function != 0 ||
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_FINAL_TOKEN_TAIL) == 0u ||
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) == 0u ||
        exact_stage_slice_plan->workspace == 0)
    {
        return false;
    }
    candidate_count =
        (uint64_t)exact_stage_slice_plan->maximum_active_sequence_count *
        (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u) *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT;
    required_workspace_bytes =
        (candidate_count * (uint64_t)sizeof(float)) +
        (candidate_count * (uint64_t)sizeof(uint32_t));
    return exact_stage_slice_plan->workspace_bytes >= required_workspace_bytes;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchBuiltInFusedFinalTokenEpilogue(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    uint64_t candidate_count;
    float *candidate_scores;
    uint32_t *candidate_tokens;
    dim3 candidate_grid;
    SparkStatus status;

    if (!SparkGlm52ResidentDecodeStageExactPlanUsesBuiltInFusedFinalTokenEpilogue(
            exact_stage_slice_plan) ||
        active_sequence_count == 0u ||
        active_sequence_count > exact_stage_slice_plan->maximum_active_sequence_count ||
        (node_context->mtp_draft_plan != 0 &&
         node_context->mtp_draft_plan->launch_function != 0) ||
        node_context->restricted_lm_head_weight_bf16 == 0 ||
        node_context->mtp_mxfp4_weight_payload_u8 == 0 ||
        node_context->mtp_mxfp4_scale_e8m0_u8 == 0 ||
        node_context->restricted_token_ids == 0 ||
        pipeline_slot->normalized_hidden_bf16 == 0 ||
        pipeline_slot->mtp_draft_hidden_bf16 == 0 ||
        pipeline_slot->mtp_target_token_ids == 0 ||
        pipeline_slot->restricted_selected_token_ids == 0 ||
        pipeline_slot->restricted_selected_token_scores == 0 ||
        pipeline_slot->mtp_draft_token_ids == 0 ||
        pipeline_slot->mtp_accept_mask == 0 ||
        pipeline_slot->mtp_committed_token_ids == 0 ||
        pipeline_slot->mtp_event_counters == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    candidate_count =
        (uint64_t)exact_stage_slice_plan->maximum_active_sequence_count *
        (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u) *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT;
    candidate_scores = (float *)exact_stage_slice_plan->workspace;
    candidate_tokens = (uint32_t *)(candidate_scores + candidate_count);
    candidate_grid = dim3(
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT);

    SparkGlm52ResidentDecodeStageFusedFinalTokenCandidateKernel<<<
        candidate_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->normalized_hidden_bf16,
        (const uint16_t *)node_context->restricted_lm_head_weight_bf16,
        (const uint16_t *)pipeline_slot->mtp_draft_hidden_bf16,
        (const uint8_t *)node_context->mtp_mxfp4_weight_payload_u8,
        (const uint8_t *)node_context->mtp_mxfp4_scale_e8m0_u8,
        node_context->restricted_token_ids,
        0,
        candidate_scores,
        candidate_tokens,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ResidentDecodeStageFusedFinalTokenCommitKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT,
        0u,
        cuda_stream>>>(
        candidate_scores,
        candidate_tokens,
        pipeline_slot->mtp_target_token_ids,
        pipeline_slot->restricted_selected_token_ids,
        pipeline_slot->restricted_selected_token_scores,
        pipeline_slot->mtp_draft_token_ids,
        pipeline_slot->mtp_accept_mask,
        pipeline_slot->mtp_committed_token_ids,
        pipeline_slot->mtp_event_counters,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    return status;
}

static SparkStatus SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
    const char *phase_name,
    SparkStatus status)
{
    if (status != SPARK_STATUS_OK &&
        getenv("GLM52_LAYER_BODY_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "layer_body_failed phase=%s status=%d\n",
            phase_name != 0 ? phase_name : "unknown",
            (int)status);
    }
    return status;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchLayerBody(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count,
    uint32_t hidden_output_only,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan)
{
    dim3 attention_grid;
    const uint16_t *final_norm_input_bf16;
    uint64_t hidden_element_count;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "submitted_phase",
            status);
    }
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "submitted_check",
            status);
    }

    if (hidden_output_only == 0u)
    {
        SparkGlm52ResidentDecodeStageClearU32Kernel<<<
            1u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT,
            0u,
            cuda_stream>>>(
            pipeline_slot->mtp_event_counters,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
                "clear_mtp_events",
                status);
        }
    }

    SparkGlm52ResidentDecodeStageRmsNormKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->input_hidden_bf16,
        (const uint16_t *)node_context->attention_norm_weight_bf16,
        (uint16_t *)pipeline_slot->normalized_hidden_bf16,
        active_sequence_count,
        node_context->rms_norm_epsilon);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "attention_norm",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "attention_norm_phase",
            status);
    }

    if (node_context->projection_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16)
    {
        status = SparkGlm52ResidentDecodeStageLaunchRawGlmProjectionWithBranchOverlap(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count,
            exact_stage_slice_plan);
    }
    else
    {
        status = SparkGlm52ResidentDecodeStageLaunchLoweredProjection(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "attention_projection",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "attention_projection_phase",
            status);
    }

    status = SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "sparse_index_selection",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "dsa_selection_phase",
            status);
    }

    SparkGlm52ResidentDecodeStagePrepareKernel<<<
        SparkGlm52ResidentDecodeStagePrepareBlockCount(active_sequence_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->query_rope_input_bf16,
        (const uint16_t *)pipeline_slot->key_rope_input_bf16,
        (const uint16_t *)pipeline_slot->current_kv_latent_bf16,
        (const uint16_t *)pipeline_slot->raw_kv_b_bf16,
        pipeline_slot->positions,
        pipeline_slot->slot_mapping,
        node_context->cos_table,
        node_context->sin_table,
        (uint16_t *)pipeline_slot->rotated_query_rope_bf16,
        (uint16_t *)node_context->mla_cache_bf16,
        (uint16_t *)node_context->key_nope_cache_bf16,
        (uint16_t *)node_context->value_cache_bf16,
        active_sequence_count,
        node_context->position_count,
        node_context->cache_token_capacity);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "rope_kv_write",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "rope_kv_write_phase",
            status);
    }

    attention_grid = dim3(
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT,
        1u);
    if (node_context->attention_execution_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX)
    {
        SparkGlm52ResidentDecodeStageAttentionOnlineKernel<<<
            attention_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->query_latent_bf16,
            (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
            (const uint16_t *)node_context->mla_cache_bf16,
            (const uint16_t *)node_context->key_nope_cache_bf16,
            (const uint16_t *)node_context->value_cache_bf16,
            pipeline_slot->block_table,
            pipeline_slot->context_lengths,
            pipeline_slot->first_block_token_offsets,
            pipeline_slot->sparse_token_indices,
            (uint16_t *)pipeline_slot->attention_output_latent_bf16,
            SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context),
            node_context->max_blocks_per_sequence,
            node_context->kv_block_count,
            node_context->cache_token_capacity,
            node_context->qk_scale);
    }
    else
    {
        SparkGlm52ResidentDecodeStageAttentionKernel<<<
            attention_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->query_latent_bf16,
            (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
            (const uint16_t *)node_context->mla_cache_bf16,
            (const uint16_t *)node_context->key_nope_cache_bf16,
            (const uint16_t *)node_context->value_cache_bf16,
            pipeline_slot->block_table,
            pipeline_slot->context_lengths,
            pipeline_slot->first_block_token_offsets,
            pipeline_slot->sparse_token_indices,
            (uint16_t *)pipeline_slot->attention_output_latent_bf16,
            SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context),
            node_context->max_blocks_per_sequence,
            node_context->kv_block_count,
            node_context->cache_token_capacity,
            node_context->qk_scale);
    }
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "mla_attention",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "mla_attention_phase",
            status);
    }

    status = SparkGlm52ResidentDecodeStageLaunchRawLinear(
        node_context,
        cuda_slot_state,
        cuda_stream,
        (const uint16_t *)pipeline_slot->attention_output_latent_bf16,
        (const uint16_t *)node_context->attention_output_weight_bf16,
        node_context->attention_output_weight_fp8_e4m3,
        node_context->attention_output_weight_scale_inv_f32,
        (uint16_t *)pipeline_slot->attention_projected_hidden_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "attention_output_projection",
            status);
    }
    hidden_element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    SparkGlm52ResidentDecodeStageResidualKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->input_hidden_bf16,
        (const uint16_t *)pipeline_slot->attention_projected_hidden_bf16,
        (uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "post_attention_residual",
            status);
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "output_projection_phase",
            status);
    }

    status = SparkGlm52ResidentDecodeStageLaunchPostAttentionMlp(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        exact_stage_slice_plan);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "post_attention_mlp",
            status);
    }

    if (hidden_output_only != 0u)
    {
        status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->layer_body_success_count += 1u;
        }
        return SPARK_STATUS_OK;
    }

    final_norm_input_bf16 = node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY
        ? (const uint16_t *)pipeline_slot->post_attention_hidden_bf16
        : (const uint16_t *)pipeline_slot->layer_output_hidden_bf16;
    SparkGlm52ResidentDecodeStageRmsNormKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        final_norm_input_bf16,
        (const uint16_t *)node_context->final_norm_weight_bf16,
        (uint16_t *)pipeline_slot->normalized_hidden_bf16,
        active_sequence_count,
        node_context->rms_norm_epsilon);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkGlm52ResidentDecodeStageExactPlanUsesBuiltInFusedFinalTokenEpilogue(
            exact_stage_slice_plan))
    {
        status = SparkGlm52ResidentDecodeStageLaunchBuiltInFusedFinalTokenEpilogue(
            exact_stage_slice_plan,
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    else
    {
        status = SparkGlm52ResidentDecodeStageLaunchRestrictedLogits(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        status = SparkGlm52ResidentDecodeStageLaunchMtpDraft(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        status = SparkGlm52ResidentDecodeStageLaunchFusedFinalTokenTail(
            exact_stage_slice_plan,
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_RESTRICTED_LOGITS);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_DRAFT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_VERIFY);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (cuda_slot_state != 0)
    {
        cuda_slot_state->layer_body_success_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Sm121RequiredDecodeStageSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state;
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    uint64_t graph_specialization_signature;
    uint32_t graph_capture_active;
    uint32_t hidden_output_only;
    bool full_stage_plan_was_launched;
    SparkStatus status;

    if (node_context == 0 ||
        (completion != 0 && completion->function == 0) ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((node_context->reserved_execution_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH) != 0u &&
        active_sequence_count != node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    runtime_pipeline_slot = node_context->pipeline_slots[pipeline_slot_index];
    if (runtime_kv_block_table != 0)
    {
        runtime_pipeline_slot.block_table =
            runtime_kv_block_table->physical_block_indices;
    }
    pipeline_slot = &runtime_pipeline_slot;
    cuda_slot_state = node_context->cuda_pipeline_slot_states != 0
        ? &node_context->cuda_pipeline_slot_states[pipeline_slot_index]
        : 0;
    cuda_stream = (cudaStream_t)pipeline_slot->cuda_stream;
    graph_capture_active = 0u;
    hidden_output_only =
        (node_context->reserved_execution_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY) != 0u;
    graph_specialization_signature =
        SparkGlm52ResidentDecodeStageComputeGraphSignature(
            node_context,
            pipeline_slot,
            hidden_output_only);
    full_stage_plan_was_launched = false;

    status = SparkGlm52ResidentDecodeStageTryLaunchFullStagePlan(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        &full_stage_plan_was_launched);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (full_stage_plan_was_launched)
    {
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            cuda_stream,
            cuda_slot_state,
            completion);
    }

    if (node_context->enable_cuda_graph_replay != 0u &&
        cuda_slot_state != 0 &&
        cuda_slot_state->cuda_graph_exec != 0 &&
        cuda_slot_state->graph_active_sequence_count == active_sequence_count &&
        cuda_slot_state->graph_specialization_signature ==
            graph_specialization_signature)
    {
        cuda_status = cudaGraphLaunch(
            (cudaGraphExec_t)cuda_slot_state->cuda_graph_exec,
            cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->graph_replay_count += 1u;
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            cuda_stream,
            cuda_slot_state,
            completion);
    }

    if (node_context->enable_cuda_graph_replay != 0u && cuda_slot_state != 0)
    {
        if (cuda_slot_state->cuda_graph_exec != 0 &&
            (cuda_slot_state->graph_active_sequence_count != active_sequence_count ||
             cuda_slot_state->graph_specialization_signature !=
                graph_specialization_signature))
        {
            cudaGraphExecDestroy((cudaGraphExec_t)cuda_slot_state->cuda_graph_exec);
            cuda_slot_state->cuda_graph_exec = 0;
            cuda_slot_state->graph_active_sequence_count = 0u;
            cuda_slot_state->graph_specialization_signature = 0u;
        }
        cuda_status = cudaStreamBeginCapture(
            cuda_stream,
            cudaStreamCaptureModeThreadLocal);
        if (cuda_status == cudaSuccess)
        {
            graph_capture_active = 1u;
        }
        else
        {
            cuda_slot_state->launch_error_count += 1u;
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (cuda_slot_state != 0)
    {
        cuda_slot_state->launch_chain_count += 1u;
    }

    status = SparkGlm52ResidentDecodeStageLaunchLayerBody(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        hidden_output_only,
        0);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (graph_capture_active != 0u && cuda_slot_state != 0)
    {
        cuda_slot_state->graph_active_sequence_count = active_sequence_count;
        cuda_slot_state->graph_capture_count += 1u;
    }
    return SparkGlm52ResidentDecodeStageFinishSubmit(
        cuda_stream,
        cuda_slot_state,
        graph_capture_active,
        graph_specialization_signature,
        completion);
}


static uint32_t SparkGlm52ResidentDecodeStageStageSliceHiddenOutputOnly(
    uint32_t layer_offset,
    uint32_t layer_count,
    uint32_t final_token_stage)
{
    return final_token_stage == 0u || layer_offset + 1u < layer_count
        ? 1u
        : 0u;
}

static SparkGlm52ResidentDecodeStageCudaPipelineSlotState *
SparkGlm52ResidentDecodeStageGetCudaSlotState(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index)
{
    if (node_context == 0 || node_context->cuda_pipeline_slot_states == 0)
    {
        return 0;
    }
    return &node_context->cuda_pipeline_slot_states[pipeline_slot_index];
}

static bool SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context);

static uint64_t SparkGlm52ResidentDecodeStageComputeStageSliceGraphSignature(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context)
{
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    uint32_t layer_offset;
    uint32_t hidden_output_only;
    uint64_t signature;
    uint64_t layer_signature;

    signature = 1469598103934665603ull;
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        0x5354414745534c43ull);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        layer_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        pipeline_slot_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        final_token_stage);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context)
            ? 1u
            : 0u);
    if (SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context))
    {
        uint32_t tap_index;

        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                frame_context->dspark_hidden_tap_plan));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            frame_context->dspark_hidden_tap_lane_stride_bytes);
        for (tap_index = 0u;
             tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
             ++tap_index)
        {
            signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                signature,
                SparkGlm52ResidentDecodeStagePointerGraphSignature(
                    frame_context->dspark_hidden_tap_output_bf16[tap_index]));
        }
    }
    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        hidden_output_only =
            SparkGlm52ResidentDecodeStageStageSliceHiddenOutputOnly(
                layer_offset,
                layer_count,
                final_token_stage);
        runtime_pipeline_slot =
            layer_node_context->pipeline_slots[pipeline_slot_index];
        if (runtime_kv_block_table != 0)
        {
            runtime_pipeline_slot.block_table =
                runtime_kv_block_table->physical_block_indices;
        }
        layer_signature = SparkGlm52ResidentDecodeStageComputeGraphSignature(
            layer_node_context,
            &runtime_pipeline_slot,
            hidden_output_only);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                layer_node_context));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            layer_signature);
    }
    return signature;
}

static void SparkGlm52ResidentDecodeStageAbortGraphCapture(
    cudaStream_t cuda_stream)
{
    cudaGraph_t abandoned_graph;

    abandoned_graph = 0;
    if (cudaStreamEndCapture(cuda_stream, &abandoned_graph) == cudaSuccess)
    {
        if (abandoned_graph != 0)
        {
            cudaGraphDestroy(abandoned_graph);
        }
    }
    else
    {
        cudaStreamSynchronize(cuda_stream);
    }
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateStageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    cudaStream_t cuda_stream)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    uint32_t layer_offset;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        (stage_slice_plan != 0 &&
         (stage_slice_plan->maximum_layer_count < layer_count ||
          stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
          (stage_slice_plan->launch_function == 0 &&
           (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_EXACT_PP13_FIXED6) == 0u))) ||
        active_sequence_count == 0u ||
        final_token_stage > 1u ||
        cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        if (layer_node_context == 0 ||
            layer_node_context->abi_version !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION ||
            layer_node_context->pipeline_slots == 0 ||
            pipeline_slot_index >= layer_node_context->pipeline_slot_count ||
            active_sequence_count >
                layer_node_context->max_active_sequence_count ||
            layer_node_context->pipeline_slots[pipeline_slot_index].cuda_stream !=
                cuda_stream ||
            layer_node_context->enable_cuda_graph_replay !=
                first_node_context->enable_cuda_graph_replay)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (stage_slice_plan == 0 &&
            (layer_node_context->full_stage_plan != 0 ||
             (layer_node_context->reserved_execution_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN) != 0u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (stage_slice_plan == 0 &&
            (layer_node_context->reserved_execution_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN) != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((layer_node_context->reserved_execution_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH) != 0u &&
            active_sequence_count != layer_node_context->max_active_sequence_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52Sm121RequiredDecodeStageQuiesceGraphs(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint32_t pipeline_slot_index;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        cudaStreamSynchronize((cudaStream_t)
            node_context->pipeline_slots[pipeline_slot_index].cuda_stream);
        if (node_context->cuda_pipeline_slot_states != 0 &&
            node_context->cuda_pipeline_slot_states[pipeline_slot_index].cuda_graph_exec != 0)
        {
            cudaGraphExecDestroy((cudaGraphExec_t)
                node_context->cuda_pipeline_slot_states[
                    pipeline_slot_index].cuda_graph_exec);
            node_context->cuda_pipeline_slot_states[
                pipeline_slot_index].cuda_graph_exec = 0;
            node_context->cuda_pipeline_slot_states[
                pipeline_slot_index].graph_active_sequence_count = 0u;
            node_context->cuda_pipeline_slot_states[
                pipeline_slot_index].graph_specialization_signature = 0u;
        }
    }
}


static SparkStatus SparkGlm52Sm121RequiredDecodeStageInitializeRequiredMoe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;
    SparkStatus status;

    if (node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        return SparkGlm52ResidentDecodeStageValidateFp8MoePlan(
            node_context,
            node_context->fp8_moe_plan);
    }
    if (node_context->layer_progression_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        return SPARK_STATUS_OK;
    }
    if (node_context->b12x_moe_dispatch_plan == 0 ||
        node_context->b12x_moe_dispatch_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageValidateB12xMoePlan(
        node_context,
        node_context->b12x_moe_dispatch_plan,
        &b12x_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (*(b12x_plan->state_cell) == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}


extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageInitialize(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context == 0 ||
        node_context->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION ||
        node_context->pipeline_slot_count == 0u ||
        node_context->pipeline_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52Sm121RequiredDecodeStageInitializeRequiredMoe(
        node_context);
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunch(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    void *cuda_stream)
{
    if (node_context == 0 ||
        pipeline_slot == 0 ||
        cuda_stream == 0 ||
        node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        pipeline_slot != &node_context->pipeline_slots[pipeline_slot_index] ||
        pipeline_slot->cuda_stream != cuda_stream)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SparkGlm52Sm121RequiredDecodeStageSubmit(
        node_context,
        pipeline_slot_index,
        active_sequence_count,
        runtime_kv_block_table,
        0);
}



static const void *SparkGlm52ResidentDecodeStageGetLayerHiddenOutput(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot)
{
    if (node_context == 0 || pipeline_slot == 0)
    {
        return 0;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY)
    {
        return pipeline_slot->post_attention_hidden_bf16;
    }
    return pipeline_slot->layer_output_hidden_bf16;
}

static bool SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context)
{
    return frame_context != 0 &&
        (frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_HIDDEN_TAPS) != 0u;
}

static bool SparkGlm52ResidentDecodeStageFindDsparkTapIndexForLayer(
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    uint32_t target_layer_index,
    uint32_t *tap_index_out)
{
    uint32_t tap_index;

    if (!SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context) ||
        frame_context->dspark_hidden_tap_plan == 0 ||
        tap_index_out == 0)
    {
        return false;
    }
    for (tap_index = 0u;
         tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        if (frame_context->dspark_hidden_tap_plan->tap_stages[
                tap_index].target_layer_index == target_layer_index)
        {
            *tap_index_out = tap_index;
            return true;
        }
    }
    return false;
}

static SparkStatus SparkGlm52ResidentDecodeStageMaybeCaptureDsparkHiddenTap(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *layer_pipeline_slot,
    uint32_t target_layer_index,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream)
{
    const void *hidden_output_bf16;
    void *tap_output_bf16;
    uint64_t hidden_row_bytes;
    uint32_t tap_index;

    if (!SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context))
    {
        return SPARK_STATUS_OK;
    }
    if (exact_stage_slice_plan == 0 ||
        layer_node_context == 0 ||
        layer_pipeline_slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStageFindDsparkTapIndexForLayer(
            frame_context,
            target_layer_index,
            &tap_index))
    {
        return SPARK_STATUS_OK;
    }
    hidden_output_bf16 = SparkGlm52ResidentDecodeStageGetLayerHiddenOutput(
        layer_node_context,
        layer_pipeline_slot);
    tap_output_bf16 = frame_context->dspark_hidden_tap_output_bf16[tap_index];
    hidden_row_bytes = (uint64_t)SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * 2ull;
    if (hidden_output_bf16 == 0 ||
        tap_output_bf16 == 0 ||
        frame_context->dspark_hidden_tap_lane_stride_bytes < hidden_row_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (cudaMemcpy2DAsync(
            tap_output_bf16,
            (size_t)frame_context->dspark_hidden_tap_lane_stride_bytes,
            hidden_output_bf16,
            (size_t)hidden_row_bytes,
            (size_t)hidden_row_bytes,
            (size_t)active_sequence_count,
            cudaMemcpyDeviceToDevice,
            cuda_stream) != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52ResidentDecodeStageStageSlicePlanIsExactPp13(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan)
{
    return stage_slice_plan != 0 &&
        (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_EXACT_PP13_FIXED6) != 0u &&
        stage_slice_plan->opaque_state != 0;
}

static const SparkGlm52ResidentDecodeStageExactStageSlicePlan *
SparkGlm52ResidentDecodeStageGetExactPp13StageSlicePlan(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan)
{
    if (!SparkGlm52ResidentDecodeStageStageSlicePlanIsExactPp13(
            stage_slice_plan))
    {
        return 0;
    }
    return (const SparkGlm52ResidentDecodeStageExactStageSlicePlan *)
        stage_slice_plan->opaque_state;
}


static bool SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsableCuda(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *router_plan;

    if (node_context == 0 || node_context->linear_plans == 0 ||
        node_context->linear_plan_count <=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS)
    {
        return false;
    }
    router_plan = &node_context->linear_plans[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS];
    return router_plan->output_is_f32 != 0u &&
        SparkGlm52ResidentDecodeStageLinearPlanKindIsProductionFast(
            router_plan->plan_kind) &&
        SparkGlm52ResidentDecodeStageLinearPlanIsUsable(
            router_plan,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->moe_expert_count,
            node_context->max_active_sequence_count);
}

static bool SparkGlm52ResidentDecodeStageLayerSupportsBuiltInFusedStageMoeCuda(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageB12xMoePlan *b12x_plan;

    if (node_context == 0)
    {
        return false;
    }
    if (node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY ||
        node_context->layer_progression_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        return true;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY)
    {
        return false;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK)
    {
        return SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsableCuda(
                node_context) &&
            SparkGlm52ResidentDecodeStageValidateB12xMoePlan(
                node_context,
                node_context->b12x_moe_dispatch_plan,
                &b12x_plan) == SPARK_STATUS_OK;
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK)
    {
        return SparkGlm52ResidentDecodeStageRouterLinearPlanIsUsableCuda(
                node_context) &&
            SparkGlm52ResidentDecodeStageValidateFp8MoePlan(
                node_context,
                node_context->fp8_moe_plan) == SPARK_STATUS_OK;
    }
    return false;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateExactPp13StageSlicePlan(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    cudaStream_t cuda_stream,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan **exact_stage_slice_plan_out)
{
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *next_layer_node_context;
    const SparkGlm52ResidentDecodeStagePipelineSlot *layer_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *next_layer_pipeline_slot;
    uint32_t required_capabilities;
    uint32_t exact_required_capabilities;
    uint32_t expected_first_layer_index;
    uint32_t layer_offset;
    bool requires_builtin_fused_stage_moe;
    SparkStatus status;

    if (exact_stage_slice_plan_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *exact_stage_slice_plan_out = 0;
    status = SparkGlm52ResidentDecodeStageValidateStageSlice(
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    exact_stage_slice_plan =
        SparkGlm52ResidentDecodeStageGetExactPp13StageSlicePlan(
            stage_slice_plan);
    if (exact_stage_slice_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_REQUIRED_CAPABILITIES;
    exact_required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_EXACT_PP13_CAPABILITIES;
    expected_first_layer_index = exact_stage_slice_plan->stage_index * 6u;
    if (stage_slice_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION ||
        stage_slice_plan->maximum_layer_count < 6u ||
        stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
        stage_slice_plan->validated_maximum_latency_ns == 0u ||
        (stage_slice_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        (stage_slice_plan->capability_flags & exact_required_capabilities) !=
            exact_required_capabilities ||
        exact_stage_slice_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_ABI_VERSION ||
        exact_stage_slice_plan->descriptor_bytes <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_DESCRIPTOR_BYTES ||
        exact_stage_slice_plan->layer_count != 6u ||
        exact_stage_slice_plan->first_layer_index != expected_first_layer_index ||
        exact_stage_slice_plan->first_layer_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
        exact_stage_slice_plan->first_layer_index + 6u >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
        exact_stage_slice_plan->stage_index >= 13u ||
        active_sequence_count > exact_stage_slice_plan->batch_bucket ||
        (exact_stage_slice_plan->batch_bucket != 16u &&
         exact_stage_slice_plan->batch_bucket != 32u &&
         exact_stage_slice_plan->batch_bucket != 64u) ||
        exact_stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
        exact_stage_slice_plan->validated_maximum_latency_ns == 0u ||
        (exact_stage_slice_plan->capability_flags & exact_required_capabilities) !=
            exact_required_capabilities ||
        layer_count != 6u ||
        final_token_stage !=
            (exact_stage_slice_plan->first_layer_index + 6u ==
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_QKV_BRANCH_OVERLAP) != 0u &&
        (exact_stage_slice_plan->query_branch_stream == 0 ||
         exact_stage_slice_plan->kv_branch_stream == 0 ||
         exact_stage_slice_plan->branch_ready_event == 0 ||
         exact_stage_slice_plan->query_branch_event == 0 ||
         exact_stage_slice_plan->kv_branch_event == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_AOT_STAGE_LAUNCH) != 0u &&
        exact_stage_slice_plan->launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) != 0u &&
        (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u &&
        ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) == 0u ||
         (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u ||
         (stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
        exact_stage_slice_plan->fused_moe_launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_FINAL_TOKEN_TAIL) != 0u &&
        exact_stage_slice_plan->final_token_launch_function == 0 &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    requires_builtin_fused_stage_moe =
        SparkGlm52ResidentDecodeStageExactPlanUsesBuiltinFusedStageMoe(
            exact_stage_slice_plan) != 0u;
    first_node_context = layer_node_contexts[0];
    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        if (layer_node_context == 0 ||
            layer_node_context->pipeline_slots == 0 ||
            pipeline_slot_index >= layer_node_context->pipeline_slot_count ||
            layer_node_context->pipeline_slots[pipeline_slot_index].cuda_stream !=
                cuda_stream ||
            layer_node_context->enable_cuda_graph_replay !=
                first_node_context->enable_cuda_graph_replay ||
            active_sequence_count > layer_node_context->max_active_sequence_count ||
            ((layer_node_context->reserved_execution_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH) != 0u &&
             active_sequence_count != layer_node_context->max_active_sequence_count))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (requires_builtin_fused_stage_moe &&
            !SparkGlm52ResidentDecodeStageLayerSupportsBuiltInFusedStageMoeCuda(
                layer_node_context))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (layer_offset + 1u < layer_count)
        {
            next_layer_node_context = layer_node_contexts[layer_offset + 1u];
            layer_pipeline_slot =
                &layer_node_context->pipeline_slots[pipeline_slot_index];
            next_layer_pipeline_slot =
                &next_layer_node_context->pipeline_slots[pipeline_slot_index];
            if ((exact_stage_slice_plan->capability_flags &
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_DEVICE_HIDDEN_HANDOFF) != 0u &&
                next_layer_pipeline_slot->input_hidden_bf16 !=
                    SparkGlm52ResidentDecodeStageGetLayerHiddenOutput(
                        layer_node_context,
                        layer_pipeline_slot))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    *exact_stage_slice_plan_out = exact_stage_slice_plan;
    return SPARK_STATUS_OK;
}

static uint64_t SparkGlm52ResidentDecodeStageComputeExactPp13StageSliceGraphSignature(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context)
{
    uint64_t signature;

    signature = SparkGlm52ResidentDecodeStageComputeStageSliceGraphSignature(
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        final_token_stage,
        runtime_kv_block_table,
        frame_context);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        0x5050313345584143ull);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(stage_slice_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(exact_stage_slice_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        exact_stage_slice_plan->stage_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        exact_stage_slice_plan->first_layer_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        exact_stage_slice_plan->layer_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        exact_stage_slice_plan->batch_bucket);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        exact_stage_slice_plan->capability_flags);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->query_branch_stream));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->kv_branch_stream));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->branch_ready_event));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->query_branch_event));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->kv_branch_event));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->launch_function));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->opaque_state));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->fused_moe_launch_function));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->fused_moe_state));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->final_token_launch_function));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            exact_stage_slice_plan->final_token_state));
    return signature;
}


typedef SparkStatus (*SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchFunction)(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    cudaStream_t cuda_stream);

typedef struct SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLauncher
{
    uint32_t stage_index;
    uint32_t batch_bucket;
    uint32_t final_token_stage;
    SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchFunction
        launch_function;
} SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLauncher;

static SparkStatus SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t layer_offset)
{
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *effective_node_context;
    SparkGlm52ResidentDecodeStageNodeContext runtime_node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *effective_pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *layer_cuda_slot_state;
    SparkStatus status;
    uint32_t hidden_output_only;

    if (layer_node_contexts == 0 || layer_offset >= 6u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    layer_node_context = layer_node_contexts[layer_offset];
    if (layer_node_context == 0 ||
        layer_node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= layer_node_context->pipeline_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    runtime_pipeline_slot = layer_node_context->pipeline_slots[pipeline_slot_index];
    if (runtime_kv_block_table != 0)
    {
        runtime_pipeline_slot.block_table =
            runtime_kv_block_table->physical_block_indices;
    }
    effective_pipeline_slot = &runtime_pipeline_slot;
    effective_node_context = layer_node_context;
    if ((exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u)
    {
        runtime_node_context = *layer_node_context;
        runtime_node_context.reserved_execution_flags |=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER;
        effective_node_context = &runtime_node_context;
    }

    layer_cuda_slot_state = SparkGlm52ResidentDecodeStageGetCudaSlotState(
        layer_node_context,
        pipeline_slot_index);
    if (layer_cuda_slot_state == 0)
    {
        layer_cuda_slot_state = first_cuda_slot_state;
    }

    hidden_output_only =
        SparkGlm52ResidentDecodeStageStageSliceHiddenOutputOnly(
            layer_offset,
            6u,
            final_token_stage);
    status = SparkGlm52ResidentDecodeStageLaunchLayerBody(
        effective_node_context,
        effective_pipeline_slot,
        layer_cuda_slot_state,
        cuda_stream,
        active_sequence_count,
        hidden_output_only,
        exact_stage_slice_plan);
    if (status != SPARK_STATUS_OK &&
        getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0)
    {
        fprintf(
            stderr,
            "exact_pp13_builtin_layer_failed offset=%u status=%d\n",
            layer_offset,
            (int)status);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageMaybeCaptureDsparkHiddenTap(
        exact_stage_slice_plan,
        frame_context,
        effective_node_context,
        effective_pipeline_slot,
        exact_stage_slice_plan->first_layer_index + layer_offset,
        active_sequence_count,
        cuda_stream);
}

template <uint32_t StageIndex, uint32_t BatchBucket, uint32_t FinalTokenStage>
static SparkStatus SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageSlice(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    cudaStream_t cuda_stream)
{
    SparkStatus status;

    if (exact_stage_slice_plan == 0 ||
        exact_stage_slice_plan->stage_index != StageIndex ||
        exact_stage_slice_plan->first_layer_index != StageIndex * 6u ||
        exact_stage_slice_plan->layer_count != 6u ||
        exact_stage_slice_plan->batch_bucket != BatchBucket ||
        exact_stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
        active_sequence_count > BatchBucket ||
        final_token_stage != FinalTokenStage)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        1u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        2u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        3u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        4u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageLayer(
        exact_stage_slice_plan,
        layer_node_contexts,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        first_cuda_slot_state,
        cuda_stream,
        5u);
}

#define SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, batch_bucket, final_token_stage) \
    { \
        stage_index, \
        batch_bucket, \
        final_token_stage, \
        SparkGlm52ResidentDecodeStageLaunchBuiltinExactPp13StageSlice< \
            stage_index, batch_bucket, final_token_stage> \
    }

static const SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLauncher
    SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchers[] =
{
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(0u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(0u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(0u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(1u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(1u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(1u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(2u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(2u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(2u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(3u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(3u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(3u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(4u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(4u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(4u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(5u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(5u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(5u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(6u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(6u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(6u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(7u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(7u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(7u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(8u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(8u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(8u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(9u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(9u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(9u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(10u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(10u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(10u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(11u, 16u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(11u, 32u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(11u, 64u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(12u, 16u, 1u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(12u, 32u, 1u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(12u, 64u, 1u)
};

#undef SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchBuiltinExactPp13StageSlicePlan(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    cudaStream_t cuda_stream,
    bool *plan_was_launched)
{
    const SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLauncher *launcher;
    size_t launcher_index;
    size_t launcher_count;
    SparkStatus status;

    if (plan_was_launched == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *plan_was_launched = false;
    if (exact_stage_slice_plan == 0 ||
        (exact_stage_slice_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT) == 0u)
    {
        return SPARK_STATUS_OK;
    }

    launcher_count = sizeof(SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchers) /
        sizeof(SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchers[0]);
    for (launcher_index = 0u; launcher_index < launcher_count; ++launcher_index)
    {
        launcher = &SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchers[launcher_index];
        if (launcher->stage_index == exact_stage_slice_plan->stage_index &&
            launcher->batch_bucket == exact_stage_slice_plan->batch_bucket &&
            launcher->final_token_stage == final_token_stage)
        {
            status = launcher->launch_function(
                exact_stage_slice_plan,
                layer_node_contexts,
                pipeline_slot_index,
                active_sequence_count,
                final_token_stage,
                runtime_kv_block_table,
                frame_context,
                first_cuda_slot_state,
                cuda_stream);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            *plan_was_launched = true;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchExactPp13StageSlicePlan(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    cudaStream_t cuda_stream,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    bool *plan_was_launched)
{
    SparkGlm52ResidentDecodeStageExactStageSliceLaunchFunction launch_function;
    SparkGlm52ResidentDecodeStageNodeContext runtime_node_contexts[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT];
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
    const SparkGlm52ResidentDecodeStageNodeContext *runtime_layer_node_contexts[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT];
    const SparkGlm52ResidentDecodeStageNodeContext *const *effective_layer_node_contexts;
    SparkStatus status;

    if (plan_was_launched != 0)
    {
        *plan_was_launched = false;
    }
    if (exact_stage_slice_plan == 0)
    {
        return SPARK_STATUS_OK;
    }
    if (exact_stage_slice_plan->launch_function == 0)
    {
        return SparkGlm52ResidentDecodeStageTryLaunchBuiltinExactPp13StageSlicePlan(
            exact_stage_slice_plan,
            layer_node_contexts,
            pipeline_slot_index,
            active_sequence_count,
            final_token_stage,
            runtime_kv_block_table,
            frame_context,
            first_cuda_slot_state,
            cuda_stream,
            plan_was_launched);
    }
    if (SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    effective_layer_node_contexts = layer_node_contexts;
    if (runtime_kv_block_table != 0)
    {
        SparkGlm52ResidentDecodeStageBuildRuntimeKvLayerContexts(
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            runtime_kv_block_table,
            runtime_node_contexts,
            runtime_pipeline_slots,
            runtime_layer_node_contexts);
        effective_layer_node_contexts = runtime_layer_node_contexts;
    }

    launch_function =
        (SparkGlm52ResidentDecodeStageExactStageSliceLaunchFunction)
            exact_stage_slice_plan->launch_function;
    status = launch_function(
        exact_stage_slice_plan,
        stage_slice_plan,
        effective_layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (plan_was_launched != 0)
    {
        *plan_was_launched = true;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchExactPp13StageSliceBody(
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state,
    cudaStream_t cuda_stream)
{
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *effective_node_context;
    SparkGlm52ResidentDecodeStageNodeContext runtime_node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *layer_pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *layer_cuda_slot_state;
    uint32_t hidden_output_only;
    uint32_t layer_offset;
    SparkStatus status;

    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        runtime_pipeline_slot =
            layer_node_context->pipeline_slots[pipeline_slot_index];
        if (runtime_kv_block_table != 0)
        {
            runtime_pipeline_slot.block_table =
                runtime_kv_block_table->physical_block_indices;
        }
        layer_pipeline_slot = &runtime_pipeline_slot;
        effective_node_context = layer_node_context;
        if ((exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE) != 0u &&
            (exact_stage_slice_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_FUSED_STAGE_MOE) != 0u)
        {
            runtime_node_context = *layer_node_context;
            runtime_node_context.reserved_execution_flags |=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER;
            effective_node_context = &runtime_node_context;
        }
        layer_cuda_slot_state = SparkGlm52ResidentDecodeStageGetCudaSlotState(
            layer_node_context,
            pipeline_slot_index);
        if (layer_cuda_slot_state == 0)
        {
            layer_cuda_slot_state = first_cuda_slot_state;
        }
        hidden_output_only =
            SparkGlm52ResidentDecodeStageStageSliceHiddenOutputOnly(
                layer_offset,
                layer_count,
                final_token_stage);
        status = SparkGlm52ResidentDecodeStageLaunchLayerBody(
            effective_node_context,
            layer_pipeline_slot,
            layer_cuda_slot_state,
            cuda_stream,
            active_sequence_count,
            hidden_output_only,
            exact_stage_slice_plan);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkGlm52ResidentDecodeStageMaybeCaptureDsparkHiddenTap(
            exact_stage_slice_plan,
            frame_context,
            effective_node_context,
            layer_pipeline_slot,
            exact_stage_slice_plan->first_layer_index + layer_offset,
            active_sequence_count,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    void *cuda_stream)
{
    const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan;
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state;
    cudaStream_t typed_cuda_stream;
    cudaError_t cuda_status;
    uint64_t graph_specialization_signature;
    uint32_t graph_capture_active;
    bool exact_stage_slice_plan_was_launched;
    SparkStatus status;

    typed_cuda_stream = (cudaStream_t)cuda_stream;
    graph_capture_active = 0u;
    status = SparkGlm52ResidentDecodeStageValidateExactPp13StageSlicePlan(
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        typed_cuda_stream,
        &exact_stage_slice_plan);
    if (status != SPARK_STATUS_OK)
    {
        if (getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0 ||
            getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "exact_pp13_validate_failed status=%d\n",
                (int)status);
        }
        return status;
    }

    first_node_context = layer_node_contexts[0];
    first_cuda_slot_state = SparkGlm52ResidentDecodeStageGetCudaSlotState(
        first_node_context,
        pipeline_slot_index);
    graph_specialization_signature =
        SparkGlm52ResidentDecodeStageComputeExactPp13StageSliceGraphSignature(
            stage_slice_plan,
            exact_stage_slice_plan,
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            final_token_stage,
            runtime_kv_block_table,
            frame_context);

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0 &&
        first_cuda_slot_state->cuda_graph_exec != 0 &&
        first_cuda_slot_state->graph_active_sequence_count ==
            active_sequence_count &&
        first_cuda_slot_state->graph_specialization_signature ==
            graph_specialization_signature)
    {
        cuda_status = cudaGraphLaunch(
            (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec,
            typed_cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            first_cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(typed_cuda_stream);
            if (getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
            {
                fprintf(
                    stderr,
                    "exact_pp13_graph_replay_failed error=%d message=%s\n",
                    (int)cuda_status,
                    cudaGetErrorString(cuda_status));
            }
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        first_cuda_slot_state->graph_replay_count += 1u;
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            typed_cuda_stream,
            first_cuda_slot_state,
            0);
    }

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0)
    {
        if (first_cuda_slot_state->cuda_graph_exec != 0 &&
            (first_cuda_slot_state->graph_active_sequence_count !=
                    active_sequence_count ||
             first_cuda_slot_state->graph_specialization_signature !=
                    graph_specialization_signature))
        {
            cudaGraphExecDestroy(
                (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec);
            first_cuda_slot_state->cuda_graph_exec = 0;
            first_cuda_slot_state->graph_active_sequence_count = 0u;
            first_cuda_slot_state->graph_specialization_signature = 0u;
        }
        cuda_status = cudaStreamBeginCapture(
            typed_cuda_stream,
            cudaStreamCaptureModeThreadLocal);
        if (cuda_status == cudaSuccess)
        {
            graph_capture_active = 1u;
        }
        else
        {
            first_cuda_slot_state->launch_error_count += 1u;
            if (getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0 ||
                getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
            {
                fprintf(
                    stderr,
                    "exact_pp13_begin_capture_failed error=%d message=%s\n",
                    (int)cuda_status,
                    cudaGetErrorString(cuda_status));
            }
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->launch_chain_count += 1u;
    }

    exact_stage_slice_plan_was_launched = false;
    status = SparkGlm52ResidentDecodeStageTryLaunchExactPp13StageSlicePlan(
        exact_stage_slice_plan,
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        frame_context,
        typed_cuda_stream,
        first_cuda_slot_state,
        &exact_stage_slice_plan_was_launched);
    if (status != SPARK_STATUS_OK)
    {
        if (getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0 ||
            getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "exact_pp13_try_launch_failed launched=%u status=%d\n",
                exact_stage_slice_plan_was_launched ? 1u : 0u,
                (int)status);
        }
        if (graph_capture_active != 0u)
        {
            SparkGlm52ResidentDecodeStageAbortGraphCapture(
                typed_cuda_stream);
        }
        return status;
    }

    if (!exact_stage_slice_plan_was_launched)
    {
        status = SparkGlm52ResidentDecodeStageLaunchExactPp13StageSliceBody(
            exact_stage_slice_plan,
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            active_sequence_count,
            final_token_stage,
            runtime_kv_block_table,
            frame_context,
            first_cuda_slot_state,
            typed_cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            if (getenv("GLM52_EXACT_PP13_DEBUG_LAUNCH_CHECK") != 0 ||
                getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
            {
                fprintf(
                    stderr,
                    "exact_pp13_body_failed status=%d\n",
                    (int)status);
            }
            if (graph_capture_active != 0u)
            {
                SparkGlm52ResidentDecodeStageAbortGraphCapture(
                    typed_cuda_stream);
            }
            return status;
        }
    }

    if (graph_capture_active != 0u && first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->graph_active_sequence_count =
            active_sequence_count;
        first_cuda_slot_state->graph_capture_count += 1u;
    }
    status = SparkGlm52ResidentDecodeStageFinishSubmit(
        typed_cuda_stream,
        first_cuda_slot_state,
        graph_capture_active,
        graph_specialization_signature,
        0);
    if (getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
    {
        fprintf(
            stderr,
            "exact_pp13_finish_submit status=%d graph_capture=%u\n",
            (int)status,
            graph_capture_active);
    }
    return status;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchStageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    void *cuda_stream)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *layer_pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *layer_cuda_slot_state;
    cudaStream_t typed_cuda_stream;
    cudaError_t cuda_status;
    uint64_t graph_specialization_signature;
    uint32_t graph_capture_active;
    uint32_t hidden_output_only;
    uint32_t layer_offset;
    bool stage_slice_plan_was_launched;
    SparkStatus status;

    typed_cuda_stream = (cudaStream_t)cuda_stream;
    status = SparkGlm52ResidentDecodeStageValidateStageSlice(
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        typed_cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    first_node_context = layer_node_contexts[0];
    first_cuda_slot_state = SparkGlm52ResidentDecodeStageGetCudaSlotState(
        first_node_context,
        pipeline_slot_index);
    graph_capture_active = 0u;
    graph_specialization_signature = 0u;

    if (stage_slice_plan != 0 &&
        stage_slice_plan->launch_function == 0 &&
        SparkGlm52ResidentDecodeStageStageSlicePlanIsExactPp13(
            stage_slice_plan))
    {
        status = SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice(
            stage_slice_plan,
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            active_sequence_count,
            final_token_stage,
            runtime_kv_block_table,
            frame_context,
            cuda_stream);
        if (getenv("GLM52_STAGE_SLICE_PLAN_DEBUG") != 0)
        {
            fprintf(
                stderr,
                "stage_slice_builtin_exact_returned status=%d\n",
                (int)status);
        }
        return status;
    }
    if (SparkGlm52ResidentDecodeStageFrameContextHasDsparkHiddenTaps(
            frame_context))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    stage_slice_plan_was_launched = false;
    status = SparkGlm52ResidentDecodeStageTryLaunchStageSlicePlan(
        stage_slice_plan,
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        final_token_stage,
        runtime_kv_block_table,
        first_cuda_slot_state,
        typed_cuda_stream,
        &stage_slice_plan_was_launched);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (stage_slice_plan_was_launched)
    {
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            typed_cuda_stream,
            first_cuda_slot_state,
            0);
    }

    graph_specialization_signature =
        SparkGlm52ResidentDecodeStageComputeStageSliceGraphSignature(
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            final_token_stage,
            runtime_kv_block_table,
            frame_context);

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0 &&
        first_cuda_slot_state->cuda_graph_exec != 0 &&
        first_cuda_slot_state->graph_active_sequence_count ==
            active_sequence_count &&
        first_cuda_slot_state->graph_specialization_signature ==
            graph_specialization_signature)
    {
        cuda_status = cudaGraphLaunch(
            (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec,
            typed_cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            first_cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(typed_cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        first_cuda_slot_state->graph_replay_count += 1u;
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            typed_cuda_stream,
            first_cuda_slot_state,
            0);
    }

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0)
    {
        if (first_cuda_slot_state->cuda_graph_exec != 0 &&
            (first_cuda_slot_state->graph_active_sequence_count !=
                    active_sequence_count ||
             first_cuda_slot_state->graph_specialization_signature !=
                    graph_specialization_signature))
        {
            cudaGraphExecDestroy(
                (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec);
            first_cuda_slot_state->cuda_graph_exec = 0;
            first_cuda_slot_state->graph_active_sequence_count = 0u;
            first_cuda_slot_state->graph_specialization_signature = 0u;
        }
        cuda_status = cudaStreamBeginCapture(
            typed_cuda_stream,
            cudaStreamCaptureModeThreadLocal);
        if (cuda_status == cudaSuccess)
        {
            graph_capture_active = 1u;
        }
        else
        {
            first_cuda_slot_state->launch_error_count += 1u;
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->launch_chain_count += 1u;
    }

    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        runtime_pipeline_slot =
            layer_node_context->pipeline_slots[pipeline_slot_index];
        if (runtime_kv_block_table != 0)
        {
            runtime_pipeline_slot.block_table =
                runtime_kv_block_table->physical_block_indices;
        }
        layer_pipeline_slot = &runtime_pipeline_slot;
        layer_cuda_slot_state =
            SparkGlm52ResidentDecodeStageGetCudaSlotState(
                layer_node_context,
                pipeline_slot_index);
        if (layer_cuda_slot_state == 0)
        {
            layer_cuda_slot_state = first_cuda_slot_state;
        }
        hidden_output_only =
            SparkGlm52ResidentDecodeStageStageSliceHiddenOutputOnly(
                layer_offset,
                layer_count,
                final_token_stage);
        status = SparkGlm52ResidentDecodeStageLaunchLayerBody(
            layer_node_context,
            layer_pipeline_slot,
            layer_cuda_slot_state,
            typed_cuda_stream,
            active_sequence_count,
            hidden_output_only,
            0);
        if (status != SPARK_STATUS_OK)
        {
            if (graph_capture_active != 0u)
            {
                SparkGlm52ResidentDecodeStageAbortGraphCapture(
                    typed_cuda_stream);
            }
            return status;
        }
    }

    if (graph_capture_active != 0u && first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->graph_active_sequence_count =
            active_sequence_count;
        first_cuda_slot_state->graph_capture_count += 1u;
    }
    return SparkGlm52ResidentDecodeStageFinishSubmit(
        typed_cuda_stream,
        first_cuda_slot_state,
        graph_capture_active,
        graph_specialization_signature,
        0);
}


static uint32_t SparkGlm52ResidentDecodeStageBulkPrefillPlanIsPagedChunk(
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan)
{
    return bulk_prefill_plan != 0 &&
        bulk_prefill_plan->opaque_state != 0 &&
        (bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PAGED_REQUIRED_CAPABILITIES) ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PAGED_REQUIRED_CAPABILITIES;
}

static __host__ __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStagePagedPrefillLaneTokenCount(
    const uint32_t *prompt_token_counts,
    uint32_t sequence_index,
    uint32_t prompt_token_count)
{
    if (prompt_token_counts == 0)
    {
        return prompt_token_count;
    }
    return prompt_token_counts[sequence_index] <= prompt_token_count
        ? prompt_token_counts[sequence_index]
        : prompt_token_count;
}

static __host__ __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStagePagedPrefillTokenStride(
    uint32_t prompt_token_stride,
    uint32_t prompt_token_count)
{
    return prompt_token_stride != 0u ? prompt_token_stride : prompt_token_count;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidatePagedChunkPrefillPlan(
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    const SparkGlm52ResidentDecodeStagePagedPrefillPlan **paged_prefill_plan_out)
{
    const SparkGlm52ResidentDecodeStagePagedPrefillPlan *paged_prefill_plan;
    uint32_t runtime_kv_table_required;
    uint32_t runtime_prefill_view_required;
    uint32_t variable_prompt_lengths_required;
    uint32_t prompt_token_stride;

    if (paged_prefill_plan_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *paged_prefill_plan_out = 0;
    if (!SparkGlm52ResidentDecodeStageBulkPrefillPlanIsPagedChunk(
            bulk_prefill_plan) ||
        node_context == 0 ||
        pipeline_slot == 0 ||
        active_sequence_count == 0u ||
        prompt_token_count == 0u ||
        prompt_token_count > UINT32_MAX - prompt_token_offset ||
        active_sequence_count > node_context->max_active_sequence_count ||
        active_sequence_count > bulk_prefill_plan->maximum_active_sequence_count ||
        prompt_token_count > bulk_prefill_plan->maximum_prompt_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    runtime_kv_table_required =
        (bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_KV_TABLE) != 0u;
    runtime_prefill_view_required =
        (bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_PREFILL_VIEW) != 0u;
    variable_prompt_lengths_required =
        (bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_VARIABLE_PROMPT_LENGTHS) != 0u;
    if (runtime_kv_table_required != 0u && runtime_kv_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_prefill_view_required != 0u && prefill_frame_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (prefill_frame_view != 0 &&
        (prefill_frame_view->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION ||
         prefill_frame_view->descriptor_bytes !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_DESCRIPTOR_BYTES ||
         prefill_frame_view->active_sequence_count != active_sequence_count ||
         prefill_frame_view->prompt_token_offset != prompt_token_offset ||
         prefill_frame_view->prompt_token_count != prompt_token_count ||
         prefill_frame_view->prompt_token_stride < prompt_token_count ||
         prefill_frame_view->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
         prefill_frame_view->reserved0 != 0u ||
         prefill_frame_view->prompt_positions == 0 ||
         prefill_frame_view->prompt_slot_mapping == 0 ||
         prefill_frame_view->prompt_context_lengths == 0 ||
         prefill_frame_view->prompt_first_block_token_offsets == 0 ||
         prefill_frame_view->prompt_token_counts == 0 ||
         prefill_frame_view->prompt_hidden_bf16 == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    paged_prefill_plan =
        (const SparkGlm52ResidentDecodeStagePagedPrefillPlan *)
            bulk_prefill_plan->opaque_state;
    if (paged_prefill_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_PLAN_ABI_VERSION ||
        paged_prefill_plan->descriptor_bytes <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_PLAN_DESCRIPTOR_BYTES ||
        paged_prefill_plan->block_token_count == 0u ||
        paged_prefill_plan->maximum_prompt_token_count < prompt_token_count ||
        paged_prefill_plan->maximum_active_sequence_count < active_sequence_count ||
        paged_prefill_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        paged_prefill_plan->cache_token_elements !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS ||
        paged_prefill_plan->prompt_positions == 0 ||
        paged_prefill_plan->prompt_slot_mapping == 0 ||
        paged_prefill_plan->prompt_context_lengths == 0 ||
        paged_prefill_plan->prompt_hidden_bf16 == 0 ||
        (paged_prefill_plan->prompt_block_table == 0 && runtime_kv_block_table == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    prompt_token_stride = paged_prefill_plan->prompt_token_stride != 0u
        ? paged_prefill_plan->prompt_token_stride
        : prompt_token_count;
    if (prompt_token_stride < prompt_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (variable_prompt_lengths_required != 0u &&
        paged_prefill_plan->prompt_token_counts == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_kv_block_table != 0 &&
        (runtime_kv_block_table->block_token_count !=
            paged_prefill_plan->block_token_count ||
         runtime_kv_block_table->lane_count < active_sequence_count ||
         runtime_kv_block_table->lane_stride == 0u ||
         runtime_kv_block_table->physical_block_indices == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (paged_prefill_plan->query_tile_token_count != 0u &&
        paged_prefill_plan->query_tile_token_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (paged_prefill_plan->key_tile_token_count != 0u &&
        paged_prefill_plan->key_tile_token_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (paged_prefill_plan->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_PAGED_ATTENTION) != 0u &&
        (paged_prefill_plan->prompt_query_latent_bf16 == 0 ||
         paged_prefill_plan->prompt_rotated_query_rope_bf16 == 0 ||
         paged_prefill_plan->prompt_attention_output_latent_bf16 == 0 ||
         node_context->mla_cache_bf16 == 0 ||
         node_context->key_nope_cache_bf16 == 0 ||
         node_context->value_cache_bf16 == 0 ||
         node_context->cache_token_capacity == 0u ||
         node_context->kv_block_count == 0u ||
         node_context->max_blocks_per_sequence == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *paged_prefill_plan_out = paged_prefill_plan;
    return SPARK_STATUS_OK;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageResolvePagedPrefillCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity)
{
    uint64_t addressed_token_index;
    uint64_t logical_block_index;
    uint64_t block_token_offset;
    uint64_t block_table_offset;
    uint32_t physical_block_index;
    uint64_t cache_slot_index;

    if (block_token_count == 0u)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    addressed_token_index =
        (uint64_t)first_block_token_offset + (uint64_t)token_index;
    logical_block_index = addressed_token_index / (uint64_t)block_token_count;
    if (logical_block_index >= (uint64_t)max_blocks_per_sequence)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    block_token_offset = addressed_token_index % (uint64_t)block_token_count;
    block_table_offset =
        ((uint64_t)sequence_index * (uint64_t)max_blocks_per_sequence) +
        logical_block_index;
    physical_block_index = block_table[block_table_offset];
    if (physical_block_index >= kv_block_count)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    cache_slot_index =
        ((uint64_t)physical_block_index * (uint64_t)block_token_count) +
        block_token_offset;
    if (cache_slot_index >= (uint64_t)cache_token_capacity ||
        cache_slot_index > (uint64_t)UINT32_MAX)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    return (uint32_t)cache_slot_index;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageWarpResolvePagedPrefillCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    uint32_t lane_index)
{
    uint32_t cache_slot_index;

    cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    if (lane_index == 0u)
    {
        cache_slot_index =
            SparkGlm52ResidentDecodeStageResolvePagedPrefillCacheSlot(
                block_table,
                sequence_index,
                token_index,
                first_block_token_offset,
                block_token_count,
                max_blocks_per_sequence,
                kv_block_count,
                cache_token_capacity);
    }
    return SparkGlm52ResidentDecodeStageWarpBroadcastU32(cache_slot_index, 0u);
}


static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStagePagedPrefillAttentionTiledKernel(
    const uint16_t *__restrict__ prompt_query_latent_bf16,
    const uint16_t *__restrict__ prompt_rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint16_t *__restrict__ key_nope_cache_bf16,
    const uint16_t *__restrict__ value_cache_bf16,
    const uint32_t *__restrict__ prompt_positions,
    const uint32_t *__restrict__ prompt_token_counts,
    const uint32_t *__restrict__ prompt_context_lengths,
    const uint32_t *__restrict__ prompt_first_block_token_offsets,
    const uint32_t *__restrict__ prompt_block_table,
    uint16_t *__restrict__ prompt_attention_output_latent_bf16,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    uint32_t prompt_token_stride,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_queries[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION];
    __shared__ float shared_tile_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS];
    __shared__ uint32_t shared_query_context_lengths[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    __shared__ uint32_t shared_query_positions[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    __shared__ uint32_t shared_query_valid[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    __shared__ uint32_t shared_maximum_context_length;
    float online_maximum[SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    float online_denominator[SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    float accumulated_value[SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS];
    uint32_t sequence_index;
    uint32_t query_tile_base;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t query_tile_index;
    uint32_t key_tile_index;
    uint32_t first_block_token_offset;
    uint32_t lane_prompt_token_count;
    uint32_t effective_prompt_token_stride;
    uint32_t dimension_index;
    uint32_t candidate_base;
    uint32_t tile_index;

    sequence_index = blockIdx.x;
    query_tile_base =
        blockIdx.y * SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
    head_index = blockIdx.z;
    if (sequence_index >= active_sequence_count ||
        head_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT)
    {
        return;
    }

    lane_index =
        threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index =
        threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    query_tile_index =
        warp_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS;
    key_tile_index =
        warp_index -
        query_tile_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS;
    effective_prompt_token_stride =
        SparkGlm52ResidentDecodeStagePagedPrefillTokenStride(
            prompt_token_stride,
            prompt_token_count);
    lane_prompt_token_count =
        SparkGlm52ResidentDecodeStagePagedPrefillLaneTokenCount(
            prompt_token_counts,
            sequence_index,
            prompt_token_count);
    first_block_token_offset = prompt_first_block_token_offsets == 0
        ? 0u
        : prompt_first_block_token_offsets[sequence_index];

    for (tile_index = threadIdx.x;
         tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
         tile_index += blockDim.x)
    {
        uint32_t prompt_token_index;
        uint64_t prompt_row_index;
        uint32_t current_position;
        uint32_t context_length;

        prompt_token_index = query_tile_base + tile_index;
        if (prompt_token_index < lane_prompt_token_count)
        {
            prompt_row_index =
                ((uint64_t)sequence_index * (uint64_t)effective_prompt_token_stride) +
                (uint64_t)prompt_token_index;
            current_position =
                prompt_token_offset + prompt_positions[prompt_row_index];
            context_length = prompt_context_lengths[sequence_index];
            if (context_length == 0u || context_length > current_position + 1u)
            {
                context_length = current_position + 1u;
            }
            shared_query_positions[tile_index] = current_position;
            shared_query_context_lengths[tile_index] = context_length;
            shared_query_valid[tile_index] = 1u;
        }
        else
        {
            shared_query_positions[tile_index] = 0u;
            shared_query_context_lengths[tile_index] = 0u;
            shared_query_valid[tile_index] = 0u;
        }
    }
    __syncthreads();

    for (tile_index = 0u;
         tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
         ++tile_index)
    {
        online_maximum[tile_index] = -FLT_MAX;
        online_denominator[tile_index] = 0.0f;
        accumulated_value[tile_index] = 0.0f;
    }

    if (threadIdx.x == 0u)
    {
        uint32_t maximum_context_length;

        maximum_context_length = 0u;
        for (tile_index = 0u;
             tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
             ++tile_index)
        {
            if (shared_query_valid[tile_index] != 0u &&
                shared_query_context_lengths[tile_index] > maximum_context_length)
            {
                maximum_context_length = shared_query_context_lengths[tile_index];
            }
        }
        shared_maximum_context_length = maximum_context_length;
    }
    __syncthreads();
    if (shared_maximum_context_length == 0u)
    {
        return;
    }

    for (tile_index = 0u;
         tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
         ++tile_index)
    {
        uint32_t prompt_token_index;
        uint64_t prompt_row_index;
        uint64_t query_row_index;

        prompt_token_index = query_tile_base + tile_index;
        if (shared_query_valid[tile_index] == 0u)
        {
            for (dimension_index = threadIdx.x;
                 dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
                 dimension_index += blockDim.x)
            {
                shared_queries[tile_index][dimension_index] = 0.0f;
            }
            continue;
        }
        prompt_row_index =
            ((uint64_t)sequence_index * (uint64_t)effective_prompt_token_stride) +
            (uint64_t)prompt_token_index;
        query_row_index =
            (prompt_row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
            (uint64_t)head_index;
        for (dimension_index = threadIdx.x;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
             dimension_index += blockDim.x)
        {
            shared_queries[tile_index][dimension_index] =
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    prompt_query_latent_bf16[
                        (query_row_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                        (uint64_t)dimension_index]);
        }
        for (dimension_index = threadIdx.x;
             dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
             dimension_index += blockDim.x)
        {
            shared_queries[tile_index]
                [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                 dimension_index] =
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    prompt_rotated_query_rope_bf16[
                        (query_row_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                        (uint64_t)dimension_index]);
        }
    }
    __syncthreads();

    dimension_index = threadIdx.x;
    for (candidate_base = 0u;
         candidate_base < shared_maximum_context_length;
         candidate_base += SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS)
    {
        uint32_t candidate_index;
        uint32_t cache_slot_index;
        float attention_score;

        if (query_tile_index == 0u &&
            key_tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS)
        {
            candidate_index = candidate_base + key_tile_index;
            cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
            if (candidate_index < shared_maximum_context_length)
            {
                cache_slot_index =
                    SparkGlm52ResidentDecodeStageWarpResolvePagedPrefillCacheSlot(
                        prompt_block_table,
                        sequence_index,
                        candidate_index,
                        first_block_token_offset,
                        block_token_count,
                        max_blocks_per_sequence,
                        kv_block_count,
                        cache_token_capacity,
                        lane_index);
            }
            if (lane_index == 0u)
            {
                shared_tile_cache_slots[key_tile_index] = cache_slot_index;
            }
        }
        __syncthreads();

        if (query_tile_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS &&
            key_tile_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS)
        {
            candidate_index = candidate_base + key_tile_index;
            cache_slot_index = shared_tile_cache_slots[key_tile_index];
            attention_score = -FLT_MAX;
            if (shared_query_valid[query_tile_index] != 0u &&
                candidate_index < shared_query_context_lengths[query_tile_index] &&
                cache_slot_index != SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                attention_score = SparkGlm52ResidentDecodeStageWarpDotProduct(
                    shared_queries[query_tile_index],
                    key_nope_cache_bf16,
                    mla_cache_bf16,
                    cache_slot_index,
                    head_index,
                    lane_index,
                    qk_scale);
            }
            if (lane_index == 0u)
            {
                shared_tile_scores[query_tile_index][key_tile_index] = attention_score;
            }
        }
        __syncthreads();

        if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
        {
            for (tile_index = 0u;
                 tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
                 ++tile_index)
            {
                uint32_t key_index;

                if (shared_query_valid[tile_index] == 0u)
                {
                    continue;
                }
                for (key_index = 0u;
                     key_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS;
                     ++key_index)
                {
                    float tile_score;
                    uint32_t tile_cache_slot;

                    tile_score = shared_tile_scores[tile_index][key_index];
                    tile_cache_slot = shared_tile_cache_slots[key_index];
                    if (tile_cache_slot !=
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT &&
                        tile_score > (-FLT_MAX * 0.5f))
                    {
                        float next_maximum;
                        float old_scale;
                        float tile_scale;
                        float value;
                        uint64_t cache_element_offset;

                        next_maximum = tile_score > online_maximum[tile_index]
                            ? tile_score
                            : online_maximum[tile_index];
                        old_scale = online_denominator[tile_index] > 0.0f
                            ? __expf(online_maximum[tile_index] - next_maximum)
                            : 0.0f;
                        tile_scale = __expf(tile_score - next_maximum);
                        cache_element_offset =
                            (((uint64_t)tile_cache_slot *
                              (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                              (uint64_t)head_index) *
                             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                            (uint64_t)dimension_index;
                        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                            value_cache_bf16[cache_element_offset]);
                        accumulated_value[tile_index] =
                            (accumulated_value[tile_index] * old_scale) +
                            (value * tile_scale);
                        online_denominator[tile_index] =
                            (online_denominator[tile_index] * old_scale) + tile_scale;
                        online_maximum[tile_index] = next_maximum;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
    {
        for (tile_index = 0u;
             tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
             ++tile_index)
        {
            uint32_t prompt_token_index;
            uint64_t prompt_row_index;
            uint64_t output_row_offset;

            if (shared_query_valid[tile_index] == 0u)
            {
                continue;
            }
            prompt_token_index = query_tile_base + tile_index;
            prompt_row_index =
                ((uint64_t)sequence_index * (uint64_t)effective_prompt_token_stride) +
                (uint64_t)prompt_token_index;
            output_row_offset =
                ((prompt_row_index *
                  (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                 (uint64_t)head_index) *
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
            prompt_attention_output_latent_bf16[
                output_row_offset + (uint64_t)dimension_index] =
                SparkGlm52ResidentDecodeStageFloatToBf16(
                    online_denominator[tile_index] > 0.0f
                        ? accumulated_value[tile_index] /
                            online_denominator[tile_index]
                        : 0.0f);
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStagePagedPrefillAttentionOnlineKernel(
    const uint16_t *__restrict__ prompt_query_latent_bf16,
    const uint16_t *__restrict__ prompt_rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint16_t *__restrict__ key_nope_cache_bf16,
    const uint16_t *__restrict__ value_cache_bf16,
    const uint32_t *__restrict__ prompt_positions,
    const uint32_t *__restrict__ prompt_context_lengths,
    const uint32_t *__restrict__ prompt_first_block_token_offsets,
    const uint32_t *__restrict__ prompt_block_table,
    uint16_t *__restrict__ prompt_attention_output_latent_bf16,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_query[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION];
    __shared__ float shared_tile_scores[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES];
    uint32_t sequence_index;
    uint32_t prompt_token_index;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    uint32_t current_position;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint64_t prompt_row_index;
    uint64_t query_row_index;
    uint64_t output_row_offset;
    uint32_t dimension_index;
    uint32_t candidate_base;
    float online_maximum;
    float online_denominator;
    float accumulated_value;

    sequence_index = blockIdx.x;
    prompt_token_index = blockIdx.y;
    head_index = blockIdx.z;
    if (sequence_index >= active_sequence_count ||
        prompt_token_index >= prompt_token_count ||
        head_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT)
    {
        return;
    }

    lane_index =
        threadIdx.x &
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index =
        threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    warp_count =
        blockDim.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    prompt_row_index =
        ((uint64_t)sequence_index * (uint64_t)prompt_token_count) +
        (uint64_t)prompt_token_index;
    current_position = prompt_token_offset + prompt_positions[prompt_row_index];
    context_length = prompt_context_lengths[sequence_index];
    if (context_length == 0u || context_length > current_position + 1u)
    {
        context_length = current_position + 1u;
    }
    first_block_token_offset = prompt_first_block_token_offsets == 0
        ? 0u
        : prompt_first_block_token_offsets[sequence_index];
    query_row_index =
        (prompt_row_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
        (uint64_t)head_index;
    output_row_offset =
        query_row_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;

    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[dimension_index] =
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                prompt_query_latent_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            dimension_index] = SparkGlm52ResidentDecodeStageBf16ToFloat(
                prompt_rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    __syncthreads();

    dimension_index = threadIdx.x;
    online_maximum = -FLT_MAX;
    online_denominator = 0.0f;
    accumulated_value = 0.0f;

    for (candidate_base = 0u;
         candidate_base < context_length;
         candidate_base += warp_count)
    {
        uint32_t candidate_index;
        uint32_t cache_slot_index;
        float attention_score;
        uint32_t tile_index;

        candidate_index = candidate_base + warp_index;
        cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
        attention_score = -FLT_MAX;
        if (candidate_index < context_length)
        {
            cache_slot_index =
                SparkGlm52ResidentDecodeStageWarpResolvePagedPrefillCacheSlot(
                    prompt_block_table,
                    sequence_index,
                    candidate_index,
                    first_block_token_offset,
                    block_token_count,
                    max_blocks_per_sequence,
                    kv_block_count,
                    cache_token_capacity,
                    lane_index);
            if (cache_slot_index !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                attention_score = SparkGlm52ResidentDecodeStageWarpDotProduct(
                    shared_query,
                    key_nope_cache_bf16,
                    mla_cache_bf16,
                    cache_slot_index,
                    head_index,
                    lane_index,
                    qk_scale);
            }
        }
        if (lane_index == 0u)
        {
            shared_tile_scores[warp_index] = attention_score;
            shared_tile_cache_slots[warp_index] = cache_slot_index;
        }
        __syncthreads();

        if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
        {
            for (tile_index = 0u; tile_index < warp_count; ++tile_index)
            {
                float tile_score;
                uint32_t tile_cache_slot;

                tile_score = shared_tile_scores[tile_index];
                tile_cache_slot = shared_tile_cache_slots[tile_index];
                if (tile_cache_slot !=
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT &&
                    tile_score > (-FLT_MAX * 0.5f))
                {
                    float next_maximum;
                    float old_scale;
                    float tile_scale;
                    float value;
                    uint64_t cache_element_offset;

                    next_maximum = tile_score > online_maximum
                        ? tile_score
                        : online_maximum;
                    old_scale = online_denominator > 0.0f
                        ? __expf(online_maximum - next_maximum)
                        : 0.0f;
                    tile_scale = __expf(tile_score - next_maximum);
                    cache_element_offset =
                        (((uint64_t)tile_cache_slot *
                          (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                        (uint64_t)dimension_index;
                    value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                        value_cache_bf16[cache_element_offset]);
                    accumulated_value =
                        (accumulated_value * old_scale) + (value * tile_scale);
                    online_denominator =
                        (online_denominator * old_scale) + tile_scale;
                    online_maximum = next_maximum;
                }
            }
        }
        __syncthreads();
    }

    if (dimension_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
    {
        prompt_attention_output_latent_bf16[
            output_row_offset + (uint64_t)dimension_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                online_denominator > 0.0f
                    ? accumulated_value / online_denominator
                    : 0.0f);
    }
}

__global__ static void SparkGlm52ResidentDecodeStagePagedPrefillCopyPromptHiddenKernel(
    const uint16_t *prompt_hidden_bf16,
    uint16_t *prompt_output_hidden_bf16,
    const uint32_t *prompt_token_counts,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    uint32_t prompt_token_stride,
    uint32_t hidden_dimension)
{
    uint64_t element_index;
    uint64_t element_count;
    uint64_t stride;
    uint32_t effective_prompt_token_stride;

    effective_prompt_token_stride =
        SparkGlm52ResidentDecodeStagePagedPrefillTokenStride(
            prompt_token_stride,
            prompt_token_count);
    element_index = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
        (uint64_t)threadIdx.x;
    element_count = (uint64_t)active_sequence_count *
        (uint64_t)effective_prompt_token_stride *
        (uint64_t)hidden_dimension;
    stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    while (element_index < element_count)
    {
        uint64_t row_element_index;
        uint32_t sequence_index;
        uint32_t token_index;
        uint32_t lane_prompt_token_count;

        row_element_index = element_index / (uint64_t)hidden_dimension;
        sequence_index =
            (uint32_t)(row_element_index / (uint64_t)effective_prompt_token_stride);
        token_index =
            (uint32_t)(row_element_index -
                (uint64_t)sequence_index * (uint64_t)effective_prompt_token_stride);
        lane_prompt_token_count =
            SparkGlm52ResidentDecodeStagePagedPrefillLaneTokenCount(
                prompt_token_counts,
                sequence_index,
                prompt_token_count);
        if (token_index < lane_prompt_token_count)
        {
            prompt_output_hidden_bf16[element_index] =
                prompt_hidden_bf16[element_index];
        }
        element_index += stride;
    }
}

__global__ static void SparkGlm52ResidentDecodeStagePagedPrefillBlockMetadataKernel(
    const uint32_t *prompt_positions,
    const uint32_t *prompt_slot_mapping,
    const uint32_t *prompt_context_lengths,
    const uint32_t *prompt_token_counts,
    const uint32_t *prompt_block_table,
    uint32_t *sparse_token_indices,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    uint32_t prompt_token_stride,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity)
{
    uint32_t token_index;
    uint32_t sequence_index;
    uint32_t local_token_index;
    uint32_t effective_prompt_token_stride;
    uint32_t lane_prompt_token_count;
    uint32_t context_length;
    uint32_t absolute_token_position;
    uint32_t cache_slot;

    effective_prompt_token_stride =
        SparkGlm52ResidentDecodeStagePagedPrefillTokenStride(
            prompt_token_stride,
            prompt_token_count);
    token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= active_sequence_count * effective_prompt_token_stride)
    {
        return;
    }

    sequence_index = token_index / effective_prompt_token_stride;
    local_token_index = token_index - sequence_index * effective_prompt_token_stride;
    lane_prompt_token_count =
        SparkGlm52ResidentDecodeStagePagedPrefillLaneTokenCount(
            prompt_token_counts,
            sequence_index,
            prompt_token_count);
    if (local_token_index >= lane_prompt_token_count)
    {
        return;
    }

    context_length = prompt_context_lengths[sequence_index];
    absolute_token_position =
        prompt_token_offset + prompt_positions[token_index];
    cache_slot = SparkGlm52ResidentDecodeStageResolvePagedPrefillCacheSlot(
        prompt_block_table,
        sequence_index,
        absolute_token_position,
        0u,
        block_token_count,
        max_blocks_per_sequence,
        kv_block_count,
        cache_token_capacity);
    if (sparse_token_indices != 0 && local_token_index == 0u)
    {
        sparse_token_indices[sequence_index] = cache_slot;
    }
    if (context_length == 0u && sparse_token_indices != 0)
    {
        sparse_token_indices[sequence_index] = prompt_slot_mapping[token_index];
    }
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchPagedChunkPrefill(
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream)
{
    const SparkGlm52ResidentDecodeStagePagedPrefillPlan *paged_prefill_plan;
    const uint32_t *effective_prompt_block_table;
    const uint32_t *effective_prompt_positions;
    const uint32_t *effective_prompt_slot_mapping;
    const uint32_t *effective_prompt_context_lengths;
    const uint32_t *effective_prompt_first_block_token_offsets;
    const uint32_t *effective_prompt_token_counts;
    const void *effective_prompt_hidden_bf16;
    const void *effective_prompt_query_latent_bf16;
    const void *effective_prompt_rotated_query_rope_bf16;
    void *effective_prompt_attention_output_latent_bf16;
    void *effective_prompt_output_hidden_bf16;
    cudaStream_t typed_cuda_stream;
    uint64_t hidden_element_count;
    uint32_t metadata_token_count;
    uint32_t effective_prompt_token_stride;
    uint32_t effective_block_token_count;
    uint32_t effective_max_blocks_per_sequence;
    uint32_t effective_kv_block_count;
    uint32_t effective_cache_token_capacity;
    SparkStatus status;

    typed_cuda_stream = (cudaStream_t)cuda_stream;
    if (typed_cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ResidentDecodeStageValidatePagedChunkPrefillPlan(
        bulk_prefill_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        prompt_token_offset,
        prompt_token_count,
        runtime_kv_block_table,
        prefill_frame_view,
        &paged_prefill_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (prefill_frame_view != 0)
    {
        effective_prompt_token_stride = prefill_frame_view->prompt_token_stride;
        effective_prompt_positions = prefill_frame_view->prompt_positions;
        effective_prompt_slot_mapping = prefill_frame_view->prompt_slot_mapping;
        effective_prompt_context_lengths =
            prefill_frame_view->prompt_context_lengths;
        effective_prompt_first_block_token_offsets =
            prefill_frame_view->prompt_first_block_token_offsets;
        effective_prompt_token_counts = prefill_frame_view->prompt_token_counts;
        effective_prompt_hidden_bf16 = prefill_frame_view->prompt_hidden_bf16;
        effective_prompt_query_latent_bf16 =
            prefill_frame_view->prompt_query_latent_bf16 != 0 ?
                prefill_frame_view->prompt_query_latent_bf16 :
                paged_prefill_plan->prompt_query_latent_bf16;
        effective_prompt_rotated_query_rope_bf16 =
            prefill_frame_view->prompt_rotated_query_rope_bf16 != 0 ?
                prefill_frame_view->prompt_rotated_query_rope_bf16 :
                paged_prefill_plan->prompt_rotated_query_rope_bf16;
        effective_prompt_attention_output_latent_bf16 =
            prefill_frame_view->prompt_attention_output_latent_bf16 != 0 ?
                prefill_frame_view->prompt_attention_output_latent_bf16 :
                paged_prefill_plan->prompt_attention_output_latent_bf16;
        effective_prompt_output_hidden_bf16 =
            prefill_frame_view->prompt_output_hidden_bf16 != 0 ?
                prefill_frame_view->prompt_output_hidden_bf16 :
                paged_prefill_plan->prompt_output_hidden_bf16;
    }
    else
    {
        effective_prompt_token_stride =
            SparkGlm52ResidentDecodeStagePagedPrefillTokenStride(
                paged_prefill_plan->prompt_token_stride,
                prompt_token_count);
        effective_prompt_positions = paged_prefill_plan->prompt_positions;
        effective_prompt_slot_mapping = paged_prefill_plan->prompt_slot_mapping;
        effective_prompt_context_lengths =
            paged_prefill_plan->prompt_context_lengths;
        effective_prompt_first_block_token_offsets =
            paged_prefill_plan->prompt_first_block_token_offsets;
        effective_prompt_token_counts = paged_prefill_plan->prompt_token_counts;
        effective_prompt_hidden_bf16 = paged_prefill_plan->prompt_hidden_bf16;
        effective_prompt_query_latent_bf16 =
            paged_prefill_plan->prompt_query_latent_bf16;
        effective_prompt_rotated_query_rope_bf16 =
            paged_prefill_plan->prompt_rotated_query_rope_bf16;
        effective_prompt_attention_output_latent_bf16 =
            paged_prefill_plan->prompt_attention_output_latent_bf16;
        effective_prompt_output_hidden_bf16 =
            paged_prefill_plan->prompt_output_hidden_bf16;
    }
    if (runtime_kv_block_table != 0)
    {
        effective_prompt_block_table = runtime_kv_block_table->physical_block_indices;
        effective_block_token_count = runtime_kv_block_table->block_token_count;
        effective_max_blocks_per_sequence = runtime_kv_block_table->lane_stride;
    }
    else
    {
        effective_prompt_block_table = paged_prefill_plan->prompt_block_table;
        effective_block_token_count = paged_prefill_plan->block_token_count;
        effective_max_blocks_per_sequence = node_context->max_blocks_per_sequence;
    }
    effective_kv_block_count = node_context->kv_block_count;
    effective_cache_token_capacity = node_context->cache_token_capacity;

    metadata_token_count = active_sequence_count * effective_prompt_token_stride;
    SparkGlm52ResidentDecodeStagePagedPrefillBlockMetadataKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(metadata_token_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        typed_cuda_stream>>>(
        effective_prompt_positions,
        effective_prompt_slot_mapping,
        effective_prompt_context_lengths,
        effective_prompt_token_counts,
        effective_prompt_block_table,
        pipeline_slot->sparse_token_indices,
        active_sequence_count,
        prompt_token_offset,
        prompt_token_count,
        effective_prompt_token_stride,
        effective_block_token_count,
        effective_max_blocks_per_sequence,
        effective_kv_block_count,
        effective_cache_token_capacity);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        0,
        typed_cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if ((bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_PAGED_ATTENTION) != 0u)
    {
        dim3 paged_attention_grid;

        paged_attention_grid.x = active_sequence_count;
        paged_attention_grid.y = (prompt_token_count +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS - 1u) /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS;
        paged_attention_grid.z = SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT;
        SparkGlm52ResidentDecodeStagePagedPrefillAttentionTiledKernel<<<
            paged_attention_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            typed_cuda_stream>>>(
            (const uint16_t *)effective_prompt_query_latent_bf16,
            (const uint16_t *)effective_prompt_rotated_query_rope_bf16,
            (const uint16_t *)node_context->mla_cache_bf16,
            (const uint16_t *)node_context->key_nope_cache_bf16,
            (const uint16_t *)node_context->value_cache_bf16,
            effective_prompt_positions,
            effective_prompt_token_counts,
            effective_prompt_context_lengths,
            effective_prompt_first_block_token_offsets,
            effective_prompt_block_table,
            (uint16_t *)effective_prompt_attention_output_latent_bf16,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            effective_prompt_token_stride,
            effective_block_token_count,
            effective_max_blocks_per_sequence,
            effective_kv_block_count,
            effective_cache_token_capacity,
            node_context->qk_scale);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            0,
            typed_cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    if (effective_prompt_output_hidden_bf16 != 0)
    {
        hidden_element_count = (uint64_t)active_sequence_count *
            (uint64_t)effective_prompt_token_stride *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
        SparkGlm52ResidentDecodeStagePagedPrefillCopyPromptHiddenKernel<<<
            SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            typed_cuda_stream>>>(
            (const uint16_t *)effective_prompt_hidden_bf16,
            (uint16_t *)effective_prompt_output_hidden_bf16,
            effective_prompt_token_counts,
            active_sequence_count,
            prompt_token_count,
            effective_prompt_token_stride,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            0,
            typed_cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream)
{
    typedef SparkStatus (*SparkGlm52ResidentDecodeStageBulkPrefillLaunchFunction)(
        const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
        const SparkGlm52ResidentDecodeStageNodeContext *node_context,
        const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
        uint32_t active_sequence_count,
        uint32_t prompt_token_offset,
        uint32_t prompt_token_count,
        void *cuda_stream);
    typedef SparkStatus (*SparkGlm52ResidentDecodeStageRuntimeKvBulkPrefillLaunchFunction)(
        const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
        const SparkGlm52ResidentDecodeStageNodeContext *node_context,
        const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
        uint32_t active_sequence_count,
        uint32_t prompt_token_offset,
        uint32_t prompt_token_count,
        const SparkGlm52KvBlockTableView *runtime_kv_block_table,
        void *cuda_stream);
    typedef SparkStatus (*SparkGlm52ResidentDecodeStageRuntimePrefillViewBulkPrefillLaunchFunction)(
        const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan,
        const SparkGlm52ResidentDecodeStageNodeContext *node_context,
        const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
        uint32_t active_sequence_count,
        uint32_t prompt_token_offset,
        uint32_t prompt_token_count,
        const SparkGlm52KvBlockTableView *runtime_kv_block_table,
        const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
        void *cuda_stream);
    SparkGlm52ResidentDecodeStageBulkPrefillLaunchFunction launch_function;
    SparkGlm52ResidentDecodeStageRuntimeKvBulkPrefillLaunchFunction runtime_kv_launch_function;
    SparkGlm52ResidentDecodeStageRuntimePrefillViewBulkPrefillLaunchFunction runtime_prefill_view_launch_function;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    const SparkGlm52ResidentDecodeStagePipelineSlot *effective_pipeline_slot;
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan;
    uint32_t required_capabilities;
    uint32_t runtime_prefill_view_required;
    SparkStatus status;

    if (node_context == 0 ||
        pipeline_slot == 0 ||
        cuda_stream == 0 ||
        node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        pipeline_slot != &node_context->pipeline_slots[pipeline_slot_index] ||
        pipeline_slot->cuda_stream != cuda_stream ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        prompt_token_count > UINT32_MAX - prompt_token_offset ||
        prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    runtime_pipeline_slot = *pipeline_slot;
    if (runtime_kv_block_table != 0)
    {
        runtime_pipeline_slot.block_table =
            runtime_kv_block_table->physical_block_indices;
    }
    effective_pipeline_slot = &runtime_pipeline_slot;

    bulk_prefill_plan = node_context->bulk_prefill_plan;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_REQUIRED_CAPABILITIES;
    runtime_prefill_view_required =
        bulk_prefill_plan != 0 &&
        (bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_PREFILL_VIEW) != 0u;
    if (bulk_prefill_plan == 0 ||
        bulk_prefill_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PLAN_ABI_VERSION ||
        active_sequence_count >
            bulk_prefill_plan->maximum_active_sequence_count ||
        prompt_token_count >
            bulk_prefill_plan->maximum_prompt_token_count ||
        (bulk_prefill_plan->launch_function == 0 &&
         !SparkGlm52ResidentDecodeStageBulkPrefillPlanIsPagedChunk(
            bulk_prefill_plan)) ||
        (bulk_prefill_plan->capability_flags & required_capabilities) !=
            required_capabilities)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_prefill_view_required != 0u && prefill_frame_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_KV_TABLE) != 0u &&
        runtime_kv_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (bulk_prefill_plan->launch_function == 0 &&
        SparkGlm52ResidentDecodeStageBulkPrefillPlanIsPagedChunk(
            bulk_prefill_plan))
    {
        status = SparkGlm52Sm121RequiredDecodeStageLaunchPagedChunkPrefill(
            bulk_prefill_plan,
            node_context,
            effective_pipeline_slot,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            runtime_kv_block_table,
            prefill_frame_view,
            cuda_stream);
    }
    else if (runtime_prefill_view_required != 0u)
    {
        runtime_prefill_view_launch_function =
            (SparkGlm52ResidentDecodeStageRuntimePrefillViewBulkPrefillLaunchFunction)
                bulk_prefill_plan->launch_function;
        status = runtime_prefill_view_launch_function(
            bulk_prefill_plan,
            node_context,
            effective_pipeline_slot,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            runtime_kv_block_table,
            prefill_frame_view,
            cuda_stream);
    }
    else if ((bulk_prefill_plan->capability_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_KV_TABLE) != 0u)
    {
        if (runtime_kv_block_table == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        runtime_kv_launch_function =
            (SparkGlm52ResidentDecodeStageRuntimeKvBulkPrefillLaunchFunction)
                bulk_prefill_plan->launch_function;
        status = runtime_kv_launch_function(
            bulk_prefill_plan,
            node_context,
            effective_pipeline_slot,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            runtime_kv_block_table,
            cuda_stream);
    }
    else
    {
        launch_function =
            (SparkGlm52ResidentDecodeStageBulkPrefillLaunchFunction)
                bulk_prefill_plan->launch_function;
        status = launch_function(
            bulk_prefill_plan,
            node_context,
            effective_pipeline_slot,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            cuda_stream);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        SparkGlm52ResidentDecodeStageGetCudaSlotState(
            node_context,
            pipeline_slot_index),
        (cudaStream_t)cuda_stream);
}

static uint64_t SparkGlm52ResidentDecodeStageComputeStageSliceBulkPrefillGraphSignature(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view)
{
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan;
    SparkGlm52ResidentDecodeStagePipelineSlot runtime_pipeline_slot;
    uint32_t layer_offset;
    uint64_t signature;
    uint64_t layer_signature;

    signature = 1469598103934665603ull;
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        0x5354414745505246ull);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        layer_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        pipeline_slot_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        prompt_token_offset);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        prompt_token_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            prefill_frame_view));
    if (prefill_frame_view != 0)
    {
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            prefill_frame_view->prompt_token_stride);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                prefill_frame_view->prompt_positions));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                prefill_frame_view->prompt_slot_mapping));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                prefill_frame_view->prompt_context_lengths));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                prefill_frame_view->prompt_token_counts));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                prefill_frame_view->prompt_hidden_bf16));
    }

    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        bulk_prefill_plan = layer_node_context->bulk_prefill_plan;
        runtime_pipeline_slot =
            layer_node_context->pipeline_slots[pipeline_slot_index];
        if (runtime_kv_block_table != 0)
        {
            runtime_pipeline_slot.block_table =
                runtime_kv_block_table->physical_block_indices;
        }
        layer_signature = SparkGlm52ResidentDecodeStageComputeGraphSignature(
            layer_node_context,
            &runtime_pipeline_slot,
            1u);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                layer_node_context));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            layer_signature);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                bulk_prefill_plan));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            bulk_prefill_plan->maximum_prompt_token_count);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            bulk_prefill_plan->capability_flags);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                bulk_prefill_plan->launch_function));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                bulk_prefill_plan->opaque_state));
    }
    return signature;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateStageSliceBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    cudaStream_t cuda_stream)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    const SparkGlm52ResidentDecodeStageBulkPrefillPlan *bulk_prefill_plan;
    uint32_t required_capabilities;
    uint32_t layer_offset;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        active_sequence_count == 0u ||
        prompt_token_count > UINT32_MAX - prompt_token_offset ||
        prompt_token_count == 0u ||
        cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_REQUIRED_CAPABILITIES;
    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        if (layer_node_context == 0 ||
            layer_node_context->abi_version !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION ||
            layer_node_context->pipeline_slots == 0 ||
            pipeline_slot_index >= layer_node_context->pipeline_slot_count ||
            active_sequence_count >
                layer_node_context->max_active_sequence_count ||
            layer_node_context->pipeline_slots[pipeline_slot_index].cuda_stream !=
                cuda_stream ||
            layer_node_context->enable_cuda_graph_replay !=
                first_node_context->enable_cuda_graph_replay)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((layer_node_context->reserved_execution_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH) != 0u &&
            active_sequence_count != layer_node_context->max_active_sequence_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        bulk_prefill_plan = layer_node_context->bulk_prefill_plan;
        if (bulk_prefill_plan == 0 ||
            bulk_prefill_plan->abi_version !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_PLAN_ABI_VERSION ||
            active_sequence_count >
                bulk_prefill_plan->maximum_active_sequence_count ||
            prompt_token_count >
                bulk_prefill_plan->maximum_prompt_token_count ||
            (bulk_prefill_plan->launch_function == 0 &&
             !SparkGlm52ResidentDecodeStageBulkPrefillPlanIsPagedChunk(
                bulk_prefill_plan)) ||
            (bulk_prefill_plan->capability_flags & required_capabilities) !=
                required_capabilities ||
            ((bulk_prefill_plan->capability_flags &
                SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_PREFILL_VIEW) != 0u &&
             prefill_frame_view == 0))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchStageSliceBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    void *cuda_stream)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *layer_node_context;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *first_cuda_slot_state;
    cudaStream_t typed_cuda_stream;
    cudaError_t cuda_status;
    uint64_t graph_specialization_signature;
    uint32_t graph_capture_active;
    uint32_t layer_offset;
    SparkStatus status;

    typed_cuda_stream = (cudaStream_t)cuda_stream;
    status = SparkGlm52ResidentDecodeStageValidateStageSliceBulkPrefill(
        layer_node_contexts,
        layer_count,
        pipeline_slot_index,
        active_sequence_count,
        prompt_token_offset,
        prompt_token_count,
        prefill_frame_view,
        typed_cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    first_node_context = layer_node_contexts[0];
    first_cuda_slot_state = SparkGlm52ResidentDecodeStageGetCudaSlotState(
        first_node_context,
        pipeline_slot_index);
    graph_capture_active = 0u;
    graph_specialization_signature =
        SparkGlm52ResidentDecodeStageComputeStageSliceBulkPrefillGraphSignature(
            layer_node_contexts,
            layer_count,
            pipeline_slot_index,
            prompt_token_offset,
            prompt_token_count,
            runtime_kv_block_table,
            prefill_frame_view);

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0 &&
        first_cuda_slot_state->cuda_graph_exec != 0 &&
        first_cuda_slot_state->graph_active_sequence_count ==
            active_sequence_count &&
        first_cuda_slot_state->graph_specialization_signature ==
            graph_specialization_signature)
    {
        cuda_status = cudaGraphLaunch(
            (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec,
            typed_cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            first_cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(typed_cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        first_cuda_slot_state->graph_replay_count += 1u;
        return SparkGlm52ResidentDecodeStageEnqueueCompletion(
            typed_cuda_stream,
            first_cuda_slot_state,
            0);
    }

    if (first_node_context->enable_cuda_graph_replay != 0u &&
        first_cuda_slot_state != 0)
    {
        if (first_cuda_slot_state->cuda_graph_exec != 0 &&
            (first_cuda_slot_state->graph_active_sequence_count !=
                    active_sequence_count ||
             first_cuda_slot_state->graph_specialization_signature !=
                    graph_specialization_signature))
        {
            cudaGraphExecDestroy(
                (cudaGraphExec_t)first_cuda_slot_state->cuda_graph_exec);
            first_cuda_slot_state->cuda_graph_exec = 0;
            first_cuda_slot_state->graph_active_sequence_count = 0u;
            first_cuda_slot_state->graph_specialization_signature = 0u;
        }
        cuda_status = cudaStreamBeginCapture(
            typed_cuda_stream,
            cudaStreamCaptureModeThreadLocal);
        if (cuda_status == cudaSuccess)
        {
            graph_capture_active = 1u;
        }
        else
        {
            first_cuda_slot_state->launch_error_count += 1u;
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->launch_chain_count += 1u;
    }

    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_node_context = layer_node_contexts[layer_offset];
        status = SparkGlm52Sm121RequiredDecodeStageLaunchBulkPrefill(
            layer_node_context,
            &layer_node_context->pipeline_slots[pipeline_slot_index],
            pipeline_slot_index,
            active_sequence_count,
            prompt_token_offset,
            prompt_token_count,
            runtime_kv_block_table,
            prefill_frame_view,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            if (graph_capture_active != 0u)
            {
                SparkGlm52ResidentDecodeStageAbortGraphCapture(
                    typed_cuda_stream);
            }
            return status;
        }
    }

    if (graph_capture_active != 0u && first_cuda_slot_state != 0)
    {
        first_cuda_slot_state->graph_active_sequence_count =
            active_sequence_count;
        first_cuda_slot_state->graph_capture_count += 1u;
    }
    return SparkGlm52ResidentDecodeStageFinishSubmit(
        typed_cuda_stream,
        first_cuda_slot_state,
        graph_capture_active,
        graph_specialization_signature,
        0);
}

extern "C" void SparkGlm52Sm121RequiredDecodeStageQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    SparkGlm52Sm121RequiredDecodeStageQuiesceGraphs(node_context);
}
