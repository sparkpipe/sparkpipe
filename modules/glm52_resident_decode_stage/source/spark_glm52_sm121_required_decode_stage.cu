#include "spark_glm52_resident_decode_stage_backend.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

#if !__has_include("flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh")
#error "FlashInfer SM120 grouped FP8 GEMM headers are required for production FP8 MoE"
#endif
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include "flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh"

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
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_WARPS 8u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_THREADS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_WARPS * 32u)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_WARPS * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE 8u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_ATTENTION_THREADS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK * 32u)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_FUSED_MAX_SCALE_BLOCKS 1024u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REFERENCE_WORKSPACE_BUFFER_COUNT 3u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES 256ull
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_CUTLASS_INT_WORKSPACE_BYTES (8ull * 1024ull * 1024ull)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_CUTLASS_FLOAT_WORKSPACE_BYTES (128ull * 1024ull * 1024ull)

#ifndef SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE
#if defined(__GNUC__) || defined(__clang__)
#define SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE __attribute__((used))
#else
#define SPARK_GLM52_REQUIRED_SYMBOL_REFERENCE
#endif
#endif

static bool SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
    const void *pointer,
    uintptr_t alignment);

static bool SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52ResidentDecodeStageQuantizedLinearView **view_out);

static uint64_t SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size);

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

static __global__ __launch_bounds__(
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_THREADS, 4)
void SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearBatchKernel(
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
    __shared__ __align__(16) __nv_bfloat16 shared_input_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K];
    __shared__ __align__(16) __nv_bfloat16 shared_weight_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K];
    __shared__ __align__(16) float shared_stage_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_WARPS *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N];
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
        float> accumulator_fragments[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N];
    uint32_t sequence_tile_begin;
    uint32_t output_tile_begin;
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t input_fill_is_vectorized;
    uint32_t input_tile_begin;
    uint32_t tile_element_index;
    uint32_t slab_offset;
    uint32_t fragment_index;

    sequence_tile_begin = blockIdx.y *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS;
    output_tile_begin = blockIdx.x *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N;
    warp_index = threadIdx.x / 32u;
    lane_index = threadIdx.x & 31u;
    input_fill_is_vectorized =
        (input_dimension &
         (SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K - 1u)) == 0u;
    for (fragment_index = 0u;
         fragment_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
         ++fragment_index)
    {
        nvcuda::wmma::fill_fragment(accumulator_fragments[fragment_index], 0.0f);
    }

    for (input_tile_begin = 0u;
         input_tile_begin < input_dimension;
         input_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K)
    {
        if (input_fill_is_vectorized != 0u)
        {
            uint32_t fill_row;
            uint32_t fill_quad;
            uint32_t sequence_index;
            uint4 packed_values;

            fill_row = threadIdx.x >> 1u;
            fill_quad = threadIdx.x & 1u;
            sequence_index = sequence_tile_begin + fill_row;
            packed_values = make_uint4(0u, 0u, 0u, 0u);
            if (sequence_index < active_sequence_count)
            {
                packed_values = *(const uint4 *)&input_bf16[
                    ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                    (uint64_t)input_tile_begin + ((uint64_t)fill_quad * 8u)];
            }
            *(uint4 *)&shared_input_tile[
                (fill_row *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K) +
                (fill_quad * 8u)] = packed_values;
        }
        else
        {
            for (tile_element_index = threadIdx.x;
                 tile_element_index <
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K;
                 tile_element_index += blockDim.x)
            {
                uint32_t local_sequence_index;
                uint32_t local_input_index;
                uint32_t sequence_index;
                uint32_t input_index;
                uint16_t input_value;

                local_sequence_index = tile_element_index /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K;
                local_input_index = tile_element_index -
                    (local_sequence_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K);
                sequence_index = sequence_tile_begin + local_sequence_index;
                input_index = input_tile_begin + local_input_index;
                input_value = 0u;
                if (sequence_index < active_sequence_count &&
                    input_index < input_dimension)
                {
                    input_value = input_bf16[
                        ((uint64_t)sequence_index * (uint64_t)input_dimension) +
                        (uint64_t)input_index];
                }
                ((uint16_t *)shared_input_tile)[tile_element_index] = input_value;
            }
        }

        for (tile_element_index = threadIdx.x;
             tile_element_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K;
             tile_element_index += blockDim.x)
        {
            uint32_t local_output_index;
            uint32_t local_input_index;
            uint32_t input_index;
            uint32_t output_index;
            float weight_value;

            local_output_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K;
            local_input_index = tile_element_index -
                (local_output_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K);
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
            shared_weight_tile[tile_element_index] = __float2bfloat16(weight_value);
        }
        __syncthreads();

        for (slab_offset = 0u;
             slab_offset < SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K;
             slab_offset += SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_K)
        {
            nvcuda::wmma::load_matrix_sync(
                input_fragment,
                shared_input_tile +
                    (warp_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K) +
                    slab_offset,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K);
#pragma unroll
            for (fragment_index = 0u;
                 fragment_index <
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
                 ++fragment_index)
            {
                nvcuda::wmma::load_matrix_sync(
                    weight_fragment,
                    shared_weight_tile +
                        (fragment_index *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K) +
                        slab_offset,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_K);
                nvcuda::wmma::mma_sync(
                    accumulator_fragments[fragment_index],
                    input_fragment,
                    weight_fragment,
                    accumulator_fragments[fragment_index]);
            }
        }
        __syncthreads();
    }

    for (fragment_index = 0u;
         fragment_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
         ++fragment_index)
    {
        nvcuda::wmma::store_matrix_sync(
            shared_stage_tile +
                (warp_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N),
            accumulator_fragments[fragment_index],
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
            nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (tile_element_index = lane_index;
             tile_element_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
             tile_element_index += 32u)
        {
            uint32_t local_row_index;
            uint32_t local_column_index;
            uint32_t sequence_index;
            uint32_t output_index;
            uint64_t output_offset;
            float output_value;

            local_row_index = tile_element_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N;
            local_column_index = tile_element_index -
                (local_row_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N);
            sequence_index = sequence_tile_begin +
                (warp_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M) +
                local_row_index;
            output_index = output_tile_begin +
                (fragment_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N) +
                local_column_index;
            if (sequence_index >= active_sequence_count ||
                output_index >= output_dimension)
            {
                continue;
            }
            output_offset =
                ((uint64_t)sequence_index * (uint64_t)output_dimension) +
                (uint64_t)output_index;
            output_value = shared_stage_tile[
                (warp_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N) +
                tile_element_index];
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
        __syncwarp();
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

static uint64_t SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size)
{
    uint64_t payload_bytes;
    uint64_t scale_block_count;
    uint64_t scale_element_count;
    uint64_t scale_bytes;
    uint64_t aligned_payload_bytes;
    uint64_t aligned_scale_bytes;

    if (maximum_active_sequence_count == 0u || input_dimension == 0u ||
        scale_block_size == 0u || (scale_block_size & 15u) != 0u)
    {
        return 0u;
    }
    if ((uint64_t)maximum_active_sequence_count >
        UINT64_MAX / (uint64_t)input_dimension)
    {
        return 0u;
    }

    payload_bytes =
        (uint64_t)maximum_active_sequence_count * (uint64_t)input_dimension;
    scale_block_count =
        ((uint64_t)input_dimension + (uint64_t)scale_block_size - 1u) /
        (uint64_t)scale_block_size;
    if (scale_block_count == 0u ||
        (uint64_t)maximum_active_sequence_count >
            UINT64_MAX / scale_block_count)
    {
        return 0u;
    }

    scale_element_count =
        (uint64_t)maximum_active_sequence_count * scale_block_count;
    if (scale_element_count > UINT64_MAX / (uint64_t)sizeof(float))
    {
        return 0u;
    }
    scale_bytes = scale_element_count * (uint64_t)sizeof(float);

    aligned_payload_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        payload_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);
    aligned_scale_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        scale_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);
    if (aligned_payload_bytes < payload_bytes ||
        aligned_scale_bytes < scale_bytes ||
        aligned_payload_bytes > UINT64_MAX - aligned_scale_bytes ||
        aligned_payload_bytes + aligned_scale_bytes >
            UINT64_MAX - aligned_scale_bytes)
    {
        return 0u;
    }
    return aligned_payload_bytes + aligned_scale_bytes + aligned_scale_bytes;
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearWorkspaceBytes(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size)
{
    return SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
        maximum_active_sequence_count,
        input_dimension,
        scale_block_size);
}

static SparkStatus SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearWorkspace(
    void *workspace,
    uint64_t workspace_bytes,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    uint8_t **activation_fp8_e4m3_out,
    float **activation_scale_f32_out,
    float **activation_amax_f32_out)
{
    uint8_t *workspace_base;
    uint64_t payload_bytes;
    uint64_t scale_block_count;
    uint64_t scale_bytes;
    uint64_t aligned_payload_bytes;
    uint64_t aligned_scale_bytes;
    uint64_t required_workspace_bytes;

    if (activation_fp8_e4m3_out == 0 || activation_scale_f32_out == 0 ||
        activation_amax_f32_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *activation_fp8_e4m3_out = 0;
    *activation_scale_f32_out = 0;
    *activation_amax_f32_out = 0;

    required_workspace_bytes =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
            maximum_active_sequence_count,
            input_dimension,
            scale_block_size);
    if (required_workspace_bytes == 0u || workspace == 0 ||
        workspace_bytes < required_workspace_bytes ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    payload_bytes =
        (uint64_t)maximum_active_sequence_count * (uint64_t)input_dimension;
    scale_block_count =
        ((uint64_t)input_dimension + (uint64_t)scale_block_size - 1u) /
        (uint64_t)scale_block_size;
    scale_bytes =
        (uint64_t)maximum_active_sequence_count * scale_block_count *
        (uint64_t)sizeof(float);
    aligned_payload_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        payload_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);
    aligned_scale_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        scale_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);

    workspace_base = (uint8_t *)workspace;
    *activation_fp8_e4m3_out = workspace_base;
    *activation_scale_f32_out =
        (float *)(workspace_base + aligned_payload_bytes);
    *activation_amax_f32_out =
        (float *)(workspace_base + aligned_payload_bytes + aligned_scale_bytes);
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageResolveFp8E4m3ActivationLinearWorkspace(
    void *workspace,
    uint64_t workspace_bytes,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    SparkGlm52Sm121RequiredDecodeStageFp8ActivationLinearWorkspaceView *workspace_view_out)
{
    SparkStatus status;
    uint8_t *activation_fp8_e4m3;
    float *activation_scale_f32;
    float *activation_amax_f32;
    uint64_t required_workspace_bytes;
    uint64_t scale_block_count;

    if (workspace_view_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(workspace_view_out, 0, sizeof(*workspace_view_out));

    required_workspace_bytes =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
            maximum_active_sequence_count,
            input_dimension,
            scale_block_size);
    if (required_workspace_bytes == 0u || scale_block_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearWorkspace(
        workspace,
        workspace_bytes,
        maximum_active_sequence_count,
        input_dimension,
        scale_block_size,
        &activation_fp8_e4m3,
        &activation_scale_f32,
        &activation_amax_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    scale_block_count =
        ((uint64_t)input_dimension + (uint64_t)scale_block_size - 1ull) /
        (uint64_t)scale_block_size;
    if (scale_block_count > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    workspace_view_out->abi_version =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_VIEW_ABI_VERSION;
    workspace_view_out->maximum_active_sequence_count = maximum_active_sequence_count;
    workspace_view_out->input_dimension = input_dimension;
    workspace_view_out->scale_block_size = scale_block_size;
    workspace_view_out->scale_block_count = (uint32_t)scale_block_count;
    workspace_view_out->required_workspace_bytes = required_workspace_bytes;
    workspace_view_out->activation_fp8_e4m3 = activation_fp8_e4m3;
    workspace_view_out->activation_scale_f32 = activation_scale_f32;
    workspace_view_out->activation_amax_f32 = activation_amax_f32;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidatePreparedFp8ActivationWeightLinearArguments(
    const uint8_t *activation_fp8_e4m3,
    const float *activation_scale_f32,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    const void *output,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    if (activation_fp8_e4m3 == 0 || activation_scale_f32 == 0 ||
        weight_fp8_e4m3 == 0 || weight_scale_inv_f32 == 0 ||
        output == 0 || cuda_stream == 0 || active_sequence_count == 0u ||
        input_dimension == 0u || output_dimension == 0u ||
        scale_block_size == 0u || scale_block_size > 1024u ||
        (scale_block_size & 15u) != 0u ||
        ((input_dimension + scale_block_size - 1u) / scale_block_size) >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_FUSED_MAX_SCALE_BLOCKS ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            activation_fp8_e4m3,
            16u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            activation_scale_f32,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            weight_fp8_e4m3,
            16u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            weight_scale_inv_f32,
            4u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}


static uint64_t SparkGlm52ResidentDecodeStageFp8ActivationLinearBackendWorkspaceOffset(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size)
{
    uint64_t activation_workspace_bytes;

    activation_workspace_bytes =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
            maximum_active_sequence_count,
            input_dimension,
            scale_block_size);
    if (activation_workspace_bytes == 0u)
    {
        return 0u;
    }
    return SparkGlm52ResidentDecodeStageAlignUpU64(
        activation_workspace_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearBackendWorkspaceBytes(
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    uint64_t backend_workspace_bytes)
{
    uint64_t backend_workspace_offset;
    uint64_t aligned_backend_workspace_bytes;

    backend_workspace_offset =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearBackendWorkspaceOffset(
            maximum_active_sequence_count,
            input_dimension,
            scale_block_size);
    if (backend_workspace_offset == 0u)
    {
        return 0u;
    }
    aligned_backend_workspace_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        backend_workspace_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES);
    if (aligned_backend_workspace_bytes < backend_workspace_bytes ||
        backend_workspace_offset > UINT64_MAX - aligned_backend_workspace_bytes)
    {
        return 0u;
    }
    return backend_workspace_offset + aligned_backend_workspace_bytes;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8ScaledGemmBackend(
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t active_sequence_count,
    uint32_t scale_block_size,
    uint32_t output_is_f32)
{
    uint32_t required_capabilities;

    if (backend == 0 || backend->launch_function == 0 ||
        backend->abi_version !=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_BACKEND_ABI_VERSION ||
        backend->cuda_architecture != 121u ||
        backend->scale_block_size != scale_block_size ||
        backend->minimum_m_alignment == 0u ||
        backend->minimum_n_alignment == 0u ||
        backend->minimum_k_alignment == 0u ||
        backend->reserved0 != 0u ||
        backend->validated_maximum_latency_ns == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_capabilities =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_REQUIRED_CAPABILITIES;
    if (output_is_f32 != 0u)
    {
        required_capabilities |=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_F32_OUTPUT;
    }
    else
    {
        required_capabilities |=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_CAPABILITY_BF16_OUTPUT;
    }
    if ((backend->capability_flags & required_capabilities) !=
            required_capabilities ||
        active_sequence_count % backend->minimum_m_alignment != 0u ||
        output_dimension % backend->minimum_n_alignment != 0u ||
        input_dimension % backend->minimum_k_alignment != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearBackendWorkspace(
    void *workspace,
    uint64_t workspace_bytes,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    uint64_t required_backend_workspace_bytes,
    uint8_t **activation_fp8_e4m3_out,
    float **activation_scale_f32_out,
    float **activation_amax_f32_out,
    void **backend_workspace_out,
    uint64_t *backend_workspace_bytes_out)
{
    uint64_t backend_workspace_offset;
    SparkStatus status;

    if (backend_workspace_out == 0 || backend_workspace_bytes_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *backend_workspace_out = 0;
    *backend_workspace_bytes_out = 0u;

    status = SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearWorkspace(
        workspace,
        workspace_bytes,
        maximum_active_sequence_count,
        input_dimension,
        scale_block_size,
        activation_fp8_e4m3_out,
        activation_scale_f32_out,
        activation_amax_f32_out);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    backend_workspace_offset =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearBackendWorkspaceOffset(
            maximum_active_sequence_count,
            input_dimension,
            scale_block_size);
    if (backend_workspace_offset == 0u ||
        workspace_bytes < backend_workspace_offset ||
        workspace_bytes - backend_workspace_offset < required_backend_workspace_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *backend_workspace_out =
        (void *)((uint8_t *)workspace + backend_workspace_offset);
    *backend_workspace_bytes_out = workspace_bytes - backend_workspace_offset;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearScaledGemmBackend(
    SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;
    uint64_t required_workspace_bytes;
    SparkStatus status;

    if (linear_plan == 0 ||
        linear_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION ||
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &quantized_view) ||
        quantized_view->weight_format !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageValidateFp8ScaledGemmBackend(
        backend,
        linear_plan->input_dimension,
        linear_plan->output_dimension,
        linear_plan->maximum_active_sequence_count,
        quantized_view->scale_block_size,
        linear_plan->output_is_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    required_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearBackendWorkspaceBytes(
            linear_plan->maximum_active_sequence_count,
            linear_plan->input_dimension,
            quantized_view->scale_block_size,
            backend->required_workspace_bytes);
    if (required_workspace_bytes == 0u || linear_plan->workspace == 0 ||
        linear_plan->workspace_bytes < required_workspace_bytes ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            linear_plan->workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    linear_plan->algorithm = (const void *)backend;
    linear_plan->custom_launch_function =
        (void *)SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationWeightLinearScaledGemmBackend(
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend,
    const void *input_bf16,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    void *workspace,
    uint64_t workspace_bytes,
    void *output,
    uint32_t active_sequence_count,
    uint32_t maximum_active_sequence_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t scale_block_size,
    uint32_t output_is_f32,
    void *cuda_stream)
{
    SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmArguments arguments;
    uint8_t *activation_fp8_e4m3;
    float *activation_scale_f32;
    float *activation_amax_f32;
    void *backend_workspace;
    uint64_t backend_workspace_bytes;
    SparkStatus status;

    if (input_bf16 == 0 || weight_fp8_e4m3 == 0 ||
        weight_scale_inv_f32 == 0 || output == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u ||
        maximum_active_sequence_count < active_sequence_count ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(weight_fp8_e4m3, 16u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(weight_scale_inv_f32, 4u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageValidateFp8ScaledGemmBackend(
        backend,
        input_dimension,
        output_dimension,
        active_sequence_count,
        scale_block_size,
        output_is_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearBackendWorkspace(
        workspace,
        workspace_bytes,
        maximum_active_sequence_count,
        input_dimension,
        scale_block_size,
        backend->required_workspace_bytes,
        &activation_fp8_e4m3,
        &activation_scale_f32,
        &activation_amax_f32,
        &backend_workspace,
        &backend_workspace_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize(
        input_bf16,
        activation_fp8_e4m3,
        activation_scale_f32,
        activation_amax_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&arguments, 0, sizeof(arguments));
    arguments.abi_version =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_ARGUMENTS_ABI_VERSION;
    arguments.active_sequence_count = active_sequence_count;
    arguments.maximum_active_sequence_count = maximum_active_sequence_count;
    arguments.input_dimension = input_dimension;
    arguments.output_dimension = output_dimension;
    arguments.scale_block_size = scale_block_size;
    arguments.output_is_f32 = output_is_f32;
    arguments.activation_scale_block_count =
        (input_dimension + scale_block_size - 1u) / scale_block_size;
    arguments.weight_input_scale_block_count = arguments.activation_scale_block_count;
    arguments.weight_output_scale_block_count =
        (output_dimension + scale_block_size - 1u) / scale_block_size;
    arguments.activation_fp8_e4m3 = activation_fp8_e4m3;
    arguments.activation_scale_f32 = activation_scale_f32;
    arguments.activation_amax_f32 = activation_amax_f32;
    arguments.weight_fp8_e4m3 = weight_fp8_e4m3;
    arguments.weight_scale_inv_f32 = weight_scale_inv_f32;
    arguments.output = output;
    arguments.workspace = backend_workspace;
    arguments.workspace_bytes = backend_workspace_bytes;
    arguments.opaque_state = backend->opaque_state;
    arguments.cuda_stream = cuda_stream;
    return backend->launch_function(&arguments);
}


static SparkStatus SparkGlm52ResidentDecodeStageLaunchFp8PreparedActivationWeightLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view,
    const uint8_t *activation_fp8_e4m3,
    const float *activation_scale_f32,
    const float *activation_amax_f32,
    void *output,
    uint32_t active_sequence_count,
    void *cuda_stream)
{
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend;
    SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmArguments arguments;
    uint64_t backend_workspace_offset;
    void *backend_workspace;
    uint64_t backend_workspace_bytes;
    SparkStatus status;

    if (linear_plan == 0 || quantized_view == 0 ||
        activation_fp8_e4m3 == 0 || activation_scale_f32 == 0 ||
        output == 0 || cuda_stream == 0 || active_sequence_count == 0u ||
        active_sequence_count > linear_plan->maximum_active_sequence_count ||
        quantized_view->weight_format !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3 ||
        quantized_view->input_dimension != linear_plan->input_dimension ||
        quantized_view->output_dimension != linear_plan->output_dimension ||
        quantized_view->weight_payload == 0 ||
        quantized_view->weight_scale == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    backend =
        (const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *)
        linear_plan->algorithm;
    if (backend == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageValidateFp8ScaledGemmBackend(
        backend,
        linear_plan->input_dimension,
        linear_plan->output_dimension,
        active_sequence_count,
        quantized_view->scale_block_size,
        linear_plan->output_is_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    backend_workspace_offset =
        SparkGlm52ResidentDecodeStageFp8ActivationLinearBackendWorkspaceOffset(
            linear_plan->maximum_active_sequence_count,
            linear_plan->input_dimension,
            quantized_view->scale_block_size);
    if (backend_workspace_offset == 0u || linear_plan->workspace == 0 ||
        linear_plan->workspace_bytes < backend_workspace_offset ||
        linear_plan->workspace_bytes - backend_workspace_offset <
            backend->required_workspace_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    backend_workspace =
        (uint8_t *)linear_plan->workspace + backend_workspace_offset;
    backend_workspace_bytes = linear_plan->workspace_bytes - backend_workspace_offset;

    memset(&arguments, 0, sizeof(arguments));
    arguments.abi_version =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_SCALED_GEMM_ARGUMENTS_ABI_VERSION;
    arguments.active_sequence_count = active_sequence_count;
    arguments.maximum_active_sequence_count = linear_plan->maximum_active_sequence_count;
    arguments.input_dimension = linear_plan->input_dimension;
    arguments.output_dimension = linear_plan->output_dimension;
    arguments.scale_block_size = quantized_view->scale_block_size;
    arguments.output_is_f32 = linear_plan->output_is_f32;
    arguments.activation_scale_block_count =
        (linear_plan->input_dimension + quantized_view->scale_block_size - 1u) /
        quantized_view->scale_block_size;
    arguments.weight_input_scale_block_count = arguments.activation_scale_block_count;
    arguments.weight_output_scale_block_count =
        (linear_plan->output_dimension + quantized_view->scale_block_size - 1u) /
        quantized_view->scale_block_size;
    arguments.activation_fp8_e4m3 = activation_fp8_e4m3;
    arguments.activation_scale_f32 = activation_scale_f32;
    arguments.activation_amax_f32 = activation_amax_f32;
    arguments.weight_fp8_e4m3 = (const uint8_t *)quantized_view->weight_payload;
    arguments.weight_scale_inv_f32 = (const float *)quantized_view->weight_scale;
    arguments.output = output;
    arguments.workspace = backend_workspace;
    arguments.workspace_bytes = backend_workspace_bytes;
    arguments.opaque_state = backend->opaque_state;
    arguments.cuda_stream = cuda_stream;
    return backend->launch_function(&arguments);
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

static __global__ void SparkGlm52ResidentDecodeStageCopySelectedTokenIndicesKernel(
    const uint32_t *__restrict__ input_indices,
    uint32_t *__restrict__ output_indices,
    uint64_t element_count)
{
    uint64_t element_index;

    element_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    if (element_index >= element_count)
    {
        return;
    }
    output_indices[element_index] = input_indices[element_index];
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
    if (lane_index <
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION / 8u))
    {
        const uint4 *lane_quad_pointer;
        uint4 packed_quad;
        uint32_t quad_pair_index;
        uint32_t query_base;

        if (lane_index <
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION / 8u))
        {
            lane_quad_pointer = ((const uint4 *)key_pairs) + lane_index;
            query_base = lane_index * 8u;
        }
        else
        {
            lane_quad_pointer = ((const uint4 *)rope_pairs) +
                (lane_index -
                 (SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION / 8u));
            query_base = lane_index * 8u;
        }
        packed_quad = *lane_quad_pointer;
        for (quad_pair_index = 0u; quad_pair_index < 4u; ++quad_pair_index)
        {
            uint32_t packed_values;

            packed_values = quad_pair_index == 0u
                ? packed_quad.x
                : quad_pair_index == 1u
                    ? packed_quad.y
                    : quad_pair_index == 2u ? packed_quad.z : packed_quad.w;
            accumulated_dot_product +=
                shared_query[query_base + (quad_pair_index * 2u)] *
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    (uint16_t)(packed_values & 0xffffu));
            accumulated_dot_product +=
                shared_query[query_base + (quad_pair_index * 2u) + 1u] *
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    (uint16_t)(packed_values >> 16u));
        }
    }
    (void)pair_index;
    return SparkGlm52ResidentDecodeStageWarpReduceSum(
        accumulated_dot_product) * qk_scale;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageAbsorbedQueryProjectKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint8_t *__restrict__ kv_b_weight_payload,
    const void *__restrict__ kv_b_weight_scale,
    uint16_t *__restrict__ query_latent_out_bf16,
    uint16_t *__restrict__ low_column_scratch_bf16,
    uint32_t active_sequence_count,
    uint32_t weight_format,
    uint32_t scale_block_size)
{
    __shared__ __nv_bfloat16 shared_weight_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    __shared__ __nv_bfloat16 shared_query_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION];
    uint32_t head_index;
    uint32_t output_tile_begin;
    uint32_t fill_index;
    uint32_t sequence_tile_begin;
    uint32_t local_sequence_index;
    uint32_t local_output_index;

    head_index = blockIdx.x;
    output_tile_begin = blockIdx.y *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE;
    for (fill_index = threadIdx.x;
         fill_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
         fill_index += blockDim.x)
    {
        uint32_t tile_output_index;
        uint32_t input_index;
        uint32_t weight_row_index;

        tile_output_index = fill_index /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
        input_index = fill_index -
            (tile_output_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION);
        weight_row_index =
            (head_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
              SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
            input_index;
        shared_weight_tile[fill_index] = __float2bfloat16(
            SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
                kv_b_weight_payload,
                kv_b_weight_scale,
                weight_format,
                scale_block_size,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                weight_row_index,
                output_tile_begin + tile_output_index));
    }
    __syncthreads();
    for (sequence_tile_begin = 0u;
         sequence_tile_begin < active_sequence_count;
         sequence_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE)
    {
        for (fill_index = threadIdx.x;
             fill_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
             fill_index += blockDim.x)
        {
            uint32_t tile_sequence_index;
            uint32_t input_index;
            uint32_t sequence_index;
            uint16_t query_value;

            tile_sequence_index = fill_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
            input_index = fill_index -
                (tile_sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION);
            sequence_index = sequence_tile_begin + tile_sequence_index;
            query_value = 0u;
            if (sequence_index < active_sequence_count)
            {
                query_value = query_latent_bf16[
                    ((((uint64_t)sequence_index *
                       (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                      (uint64_t)head_index) *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)input_index];
            }
            ((uint16_t *)shared_query_tile)[fill_index] = query_value;
        }
        __syncthreads();
        local_sequence_index = threadIdx.x /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE;
        local_output_index = threadIdx.x -
            (local_sequence_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE);
        if (sequence_tile_begin + local_sequence_index < active_sequence_count)
        {
            uint32_t input_index;
            uint32_t output_index;
            uint64_t query_row_index;
            float accumulated_value;

            accumulated_value = 0.0f;
            for (input_index = 0u;
                 input_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
                 ++input_index)
            {
                accumulated_value +=
                    SparkGlm52ResidentDecodeStageBf16ToFloat(
                        ((const uint16_t *)shared_query_tile)[
                            (local_sequence_index *
                             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                            input_index]) *
                    __bfloat162float(shared_weight_tile[
                        (local_output_index *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                        input_index]);
            }
            output_index = output_tile_begin + local_output_index;
            query_row_index =
                (((uint64_t)(sequence_tile_begin + local_sequence_index) *
                  (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                 (uint64_t)head_index);
            if (output_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)
            {
                low_column_scratch_bf16[
                    (query_row_index *
                     (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
                    (uint64_t)output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
            }
            else
            {
                query_latent_out_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
            }
        }
        __syncthreads();
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageAbsorbedQueryCommitKernel(
    const uint16_t *__restrict__ low_column_scratch_bf16,
    uint16_t *__restrict__ query_latent_bf16,
    uint32_t active_sequence_count)
{
    uint64_t element_index;
    uint64_t element_count;
    uint64_t row_index;
    uint64_t column_index;

    element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
    element_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (element_index < element_count)
    {
        row_index = element_index /
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
        column_index = element_index -
            (row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION);
        query_latent_bf16[
            (row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
            column_index] = low_column_scratch_bf16[
            (row_index *
             (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
            column_index];
        element_index += (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_ATTENTION_THREADS, 2)
void SparkGlm52ResidentDecodeStageAbsorbedAttentionKernel(
    uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ __nv_bfloat16 shared_query[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];
    __shared__ __nv_bfloat16 shared_latent_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE];
    uint32_t sequence_index;
    uint32_t head_group_begin;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t head_index;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint32_t fill_index;
    uint32_t candidate_base;
    uint32_t accumulator_index;
    uint64_t query_row_index;
    uint64_t sparse_row_offset;
    float online_maximum;
    float online_denominator;
    float accumulated_latent[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES];

    sequence_index = blockIdx.x;
    head_group_begin = blockIdx.y *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK;
    lane_index = threadIdx.x & (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index = threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
    head_index = head_group_begin + warp_index;
    context_length = context_lengths[sequence_index];
    first_block_token_offset = first_block_token_offsets[sequence_index];
    query_row_index =
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
        (uint64_t)head_index;
    sparse_row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    for (fill_index = threadIdx.x;
         fill_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
         fill_index += blockDim.x)
    {
        uint32_t local_head_index;
        uint32_t dimension_index;
        uint64_t fill_row_index;
        uint16_t query_value;

        local_head_index = fill_index /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
        dimension_index = fill_index -
            (local_head_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS);
        fill_row_index =
            ((uint64_t)sequence_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
            (uint64_t)(head_group_begin + local_head_index);
        query_value = dimension_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION
            ? query_latent_bf16[
                (fill_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                (uint64_t)dimension_index]
            : rotated_query_rope_bf16[
                (fill_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                (uint64_t)(dimension_index -
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)];
        ((uint16_t *)shared_query)[fill_index] = query_value;
    }
    online_maximum = -FLT_MAX;
    online_denominator = 0.0f;
    for (accumulator_index = 0u;
         accumulator_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
         ++accumulator_index)
    {
        accumulated_latent[accumulator_index] = 0.0f;
    }
    __syncthreads();
    for (candidate_base = 0u;
         candidate_base < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         candidate_base += SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE)
    {
        uint32_t tile_index;
        float tile_scores[SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE];
        float tile_maximum;
        float next_maximum;
        float rescale;

        if (warp_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE)
        {
            uint32_t candidate_index;
            uint32_t token_index;
            uint32_t cache_slot_index;

            candidate_index = candidate_base + warp_index;
            cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
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
            if (lane_index == 0u)
            {
                shared_tile_cache_slots[warp_index] = cache_slot_index;
            }
        }
        __syncthreads();
        for (fill_index = threadIdx.x;
             fill_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
             fill_index += blockDim.x)
        {
            uint32_t tile_slot_index;
            uint32_t dimension_index;
            uint32_t cache_slot_index;
            uint16_t latent_value;

            tile_slot_index = fill_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
            dimension_index = fill_index -
                (tile_slot_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS);
            cache_slot_index = shared_tile_cache_slots[tile_slot_index];
            latent_value = 0u;
            if (cache_slot_index !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                latent_value = mla_cache_bf16[
                    ((uint64_t)cache_slot_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    (uint64_t)dimension_index];
            }
            ((uint16_t *)shared_latent_tile)[fill_index] = latent_value;
        }
        __syncthreads();
        tile_maximum = -FLT_MAX;
        for (tile_index = 0u;
             tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE;
             ++tile_index)
        {
            uint32_t dimension_index;
            float lane_partial;
            float score;

            lane_partial = 0.0f;
            if (shared_tile_cache_slots[tile_index] !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                for (dimension_index = lane_index;
                     dimension_index <
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
                     dimension_index += SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
                {
                    lane_partial +=
                        __bfloat162float(shared_query[
                            (warp_index *
                             SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                            dimension_index]) *
                        __bfloat162float(shared_latent_tile[
                            (tile_index *
                             SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                            dimension_index]);
                }
                score = SparkGlm52ResidentDecodeStageWarpReduceSum(lane_partial) *
                    qk_scale;
            }
            else
            {
                score = -FLT_MAX;
            }
            tile_scores[tile_index] = score;
            if (score > tile_maximum)
            {
                tile_maximum = score;
            }
        }
        if (tile_maximum > (-FLT_MAX * 0.5f))
        {
            next_maximum = tile_maximum > online_maximum
                ? tile_maximum
                : online_maximum;
            rescale = online_denominator > 0.0f
                ? __expf(online_maximum - next_maximum)
                : 0.0f;
            online_denominator *= rescale;
            for (accumulator_index = 0u;
                 accumulator_index <
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
                 ++accumulator_index)
            {
                accumulated_latent[accumulator_index] *= rescale;
            }
            for (tile_index = 0u;
                 tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SLOT_TILE;
                 ++tile_index)
            {
                float probability;

                if (tile_scores[tile_index] <= (-FLT_MAX * 0.5f))
                    continue;
                probability = __expf(tile_scores[tile_index] - next_maximum);
                online_denominator += probability;
                for (accumulator_index = 0u;
                     accumulator_index <
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
                     ++accumulator_index)
                {
                    accumulated_latent[accumulator_index] += probability *
                        __bfloat162float(shared_latent_tile[
                            (tile_index *
                             SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                            (accumulator_index *
                             SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES) +
                            lane_index]);
                }
            }
            online_maximum = next_maximum;
        }
        __syncthreads();
    }
    for (accumulator_index = 0u;
         accumulator_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
         ++accumulator_index)
    {
        query_latent_bf16[
            (query_row_index *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
            (uint64_t)((accumulator_index *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES) + lane_index)] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                online_denominator > 0.0f
                    ? accumulated_latent[accumulator_index] / online_denominator
                    : 0.0f);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageAbsorbedValueApplyKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint8_t *__restrict__ kv_b_weight_payload,
    const void *__restrict__ kv_b_weight_scale,
    uint16_t *__restrict__ output_value_bf16,
    uint32_t active_sequence_count,
    uint32_t weight_format,
    uint32_t scale_block_size)
{
    __shared__ __nv_bfloat16 shared_weight_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION];
    __shared__ __nv_bfloat16 shared_latent_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION];
    uint32_t head_index;
    uint32_t output_tile_begin;
    uint32_t fill_index;
    uint32_t sequence_tile_begin;
    uint32_t local_sequence_index;
    uint32_t local_output_index;

    head_index = blockIdx.x;
    output_tile_begin = blockIdx.y *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE;
    for (fill_index = threadIdx.x;
         fill_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
         fill_index += blockDim.x)
    {
        uint32_t tile_output_index;
        uint32_t input_index;
        uint32_t weight_row_index;

        tile_output_index = fill_index /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
        input_index = fill_index -
            (tile_output_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
        weight_row_index =
            (head_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
              SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)) +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            output_tile_begin + tile_output_index;
        shared_weight_tile[fill_index] = __float2bfloat16(
            SparkGlm52ResidentDecodeStageQuantizedLinearWeightToFloat(
                kv_b_weight_payload,
                kv_b_weight_scale,
                weight_format,
                scale_block_size,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
                weight_row_index,
                input_index));
    }
    __syncthreads();
    for (sequence_tile_begin = 0u;
         sequence_tile_begin < active_sequence_count;
         sequence_tile_begin += SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE)
    {
        for (fill_index = threadIdx.x;
             fill_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_SEQ_TILE *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
             fill_index += blockDim.x)
        {
            uint32_t tile_sequence_index;
            uint32_t input_index;
            uint32_t sequence_index;
            uint16_t latent_value;

            tile_sequence_index = fill_index /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
            input_index = fill_index -
                (tile_sequence_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION);
            sequence_index = sequence_tile_begin + tile_sequence_index;
            latent_value = 0u;
            if (sequence_index < active_sequence_count)
            {
                latent_value = query_latent_bf16[
                    ((((uint64_t)sequence_index *
                       (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                      (uint64_t)head_index) *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)input_index];
            }
            ((uint16_t *)shared_latent_tile)[fill_index] = latent_value;
        }
        __syncthreads();
        local_sequence_index = threadIdx.x /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE;
        local_output_index = threadIdx.x -
            (local_sequence_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE);
        if (sequence_tile_begin + local_sequence_index < active_sequence_count)
        {
            uint32_t input_index;
            uint64_t output_row_index;
            float accumulated_value;

            accumulated_value = 0.0f;
            for (input_index = 0u;
                 input_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
                 ++input_index)
            {
                accumulated_value +=
                    __bfloat162float(shared_latent_tile[
                        (local_sequence_index *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                        input_index]) *
                    __bfloat162float(shared_weight_tile[
                        (local_output_index *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                        input_index]);
            }
            output_row_index =
                (((uint64_t)(sequence_tile_begin + local_sequence_index) *
                  (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                 (uint64_t)head_index);
            output_value_bf16[
                (output_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                (uint64_t)(output_tile_begin + local_output_index)] =
                SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
        }
        __syncthreads();
    }
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
    if (row_exponential_sum > 0.0f)
    {
        for (candidate_index = threadIdx.x;
             candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
             candidate_index += blockDim.x)
        {
            shared_scores[candidate_index] /= row_exponential_sum;
        }
    }
    __syncthreads();

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
                        shared_scores[candidate_index] *
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

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageBenchmarkForceExpertCoverageKernel(
    uint32_t *__restrict__ topk_expert_ids,
    float *__restrict__ topk_weights,
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k,
    uint32_t forced_expert_count,
    float routed_scaling_factor)
{
    uint64_t route_index;
    uint64_t route_count;
    float route_weight;

    route_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    route_count = (uint64_t)active_sequence_count * (uint64_t)top_k;
    if (topk_expert_ids == 0 || topk_weights == 0 || route_index >= route_count ||
        expert_count == 0u || top_k == 0u || forced_expert_count == 0u)
    {
        return;
    }
    if (forced_expert_count > expert_count)
    {
        forced_expert_count = expert_count;
    }
    topk_expert_ids[route_index] = (uint32_t)(route_index % forced_expert_count);
    route_weight = routed_scaling_factor / (float)top_k;
    topk_weights[route_index] = route_weight;
}

static uint32_t SparkGlm52ResidentDecodeStageBenchmarkForcedExpertCoverage(
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k)
{
    const char *text;
    char *end_pointer;
    unsigned long value;
    uint64_t route_count;

    text = getenv("GLM52_BENCHMARK_FORCE_EXPERT_COVERAGE");
    if (text == 0 || text[0] == '\0' || active_sequence_count == 0u ||
        expert_count == 0u || top_k == 0u)
    {
        return 0u;
    }
    value = strtoul(text, &end_pointer, 10);
    if (end_pointer == text || value == 0ul)
    {
        return 0u;
    }
    route_count = (uint64_t)active_sequence_count * (uint64_t)top_k;
    if (route_count > (uint64_t)UINT32_MAX)
    {
        route_count = (uint64_t)UINT32_MAX;
    }
    if (value > (unsigned long)expert_count)
    {
        value = (unsigned long)expert_count;
    }
    if (value > (unsigned long)route_count)
    {
        value = (unsigned long)route_count;
    }
    return (uint32_t)value;
}

static SparkStatus SparkGlm52ResidentDecodeStageMaybeForceBenchmarkExpertCoverage(
    uint32_t *topk_expert_ids,
    float *topk_weights,
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k,
    float routed_scaling_factor,
    cudaStream_t cuda_stream,
    const char *path_name)
{
    cudaError_t cuda_status;
    uint32_t forced_expert_count;
    uint64_t route_count;

    forced_expert_count = SparkGlm52ResidentDecodeStageBenchmarkForcedExpertCoverage(
        active_sequence_count,
        expert_count,
        top_k);
    if (forced_expert_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    route_count = (uint64_t)active_sequence_count * (uint64_t)top_k;
    SparkGlm52ResidentDecodeStageBenchmarkForceExpertCoverageKernel<<<
        (uint32_t)((route_count +
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        topk_expert_ids,
        topk_weights,
        active_sequence_count,
        expert_count,
        top_k,
        forced_expert_count,
        routed_scaling_factor);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    fprintf(
        stderr,
        "benchmark_forced_expert_coverage path=%s tokens=%u active_experts=%u of=%u routes=%llu\n",
        path_name != 0 ? path_name : "unknown",
        active_sequence_count,
        forced_expert_count,
        expert_count,
        (unsigned long long)route_count);
    return SPARK_STATUS_OK;
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

static uint32_t SparkGlm52ResidentDecodeStageQuantizedTensorCorePlanWeightFormat(
    uint32_t plan_kind)
{
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
    }
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_NVFP4_E2M1;
    }
    if (plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_MXFP4_E2M1;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16;
}

static uint32_t SparkGlm52ResidentDecodeStageQuantizedTensorCorePlanScaleBlockSize(
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

    expected_weight_format = SparkGlm52ResidentDecodeStageQuantizedTensorCorePlanWeightFormat(
        linear_plan->plan_kind);
    expected_scale_block_size = SparkGlm52ResidentDecodeStageQuantizedTensorCorePlanScaleBlockSize(
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

    if (quantized_view->weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *backend;

        backend =
            (const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *)
            linear_plan->algorithm;
        if (backend != 0)
        {
            return SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationWeightLinearScaledGemmBackend(
                backend,
                input,
                (const uint8_t *)quantized_view->weight_payload,
                (const float *)quantized_view->weight_scale,
                linear_plan->workspace,
                linear_plan->workspace_bytes,
                output,
                active_sequence_count,
                linear_plan->maximum_active_sequence_count,
                linear_plan->input_dimension,
                linear_plan->output_dimension,
                quantized_view->scale_block_size,
                linear_plan->output_is_f32,
                (void *)cuda_stream);
        }
    }

    if (active_sequence_count >
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_M)
    {
        grid = dim3(
            (linear_plan->output_dimension +
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N - 1u) /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_N,
            (active_sequence_count +
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS - 1u) /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_ROWS,
            1u);
        SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearBatchKernel<<<
            grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_BATCH_THREADS,
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
        grid = dim3(
            (linear_plan->output_dimension +
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N - 1u) /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_SUPPORTED_QKVO_WMMA_N,
            1u,
            1u);
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
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;

    if (!SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            linear_plan == 0 ? 0u : linear_plan->plan_kind) ||
        !SparkGlm52ResidentDecodeStageBlackwellQuantizedTensorCorePlanShapeIsSupported(
            linear_plan) ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &quantized_view))
    {
        return 0u;
    }
    if (quantized_view->weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        return SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
            linear_plan->maximum_active_sequence_count,
            linear_plan->input_dimension,
            quantized_view->scale_block_size);
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

    if (quantized_view->weight_format ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3)
    {
        uint64_t required_workspace_bytes;

        required_workspace_bytes =
            SparkGlm52ResidentDecodeStageFp8ActivationLinearWorkspaceBytesForShape(
                linear_plan->maximum_active_sequence_count,
                linear_plan->input_dimension,
                quantized_view->scale_block_size);
        if (required_workspace_bytes == 0u ||
            linear_plan->workspace == 0 ||
            linear_plan->workspace_bytes < required_workspace_bytes ||
            !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
                linear_plan->workspace,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_ACTIVATION_LINEAR_WORKSPACE_ALIGNMENT_BYTES))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

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
        if (SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
                linear_plan->plan_kind))
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

static uint32_t SparkGlm52Sm121RequiredDecodeStageLinearPlanIndexIsRegularDenseProjection(
    uint32_t plan_index)
{
    return SparkGlm52Sm121RequiredDecodeStageLinearPlanIndexIsRawProjection(
            plan_index) ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP ||
        plan_index == SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(
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

        if (!SparkGlm52Sm121RequiredDecodeStageLinearPlanIndexIsRegularDenseProjection(
                plan_index))
        {
            continue;
        }
        linear_plan = &linear_plans[plan_index];
        if (SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
                linear_plan->plan_kind))
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

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageFp8E4m3QuantizeRowsKernel(
    const uint16_t *__restrict__ input_bf16,
    uint8_t *__restrict__ output_fp8_e4m3,
    float *__restrict__ output_scale_f32,
    float *__restrict__ output_amax_f32,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[256u];
    uint32_t row_index;
    uint32_t scale_block_index;
    uint32_t input_begin;
    uint32_t input_end;
    uint32_t element_index;
    uint32_t scale_block_count;
    float local_absmax;
    float block_absmax;
    float scale_value;

    row_index = blockIdx.x;
    scale_block_index = blockIdx.y;
    if (row_index >= row_count || scale_block_size == 0u)
    {
        return;
    }

    input_begin = scale_block_index * scale_block_size;
    input_end = input_begin + scale_block_size;
    if (input_begin >= element_count)
    {
        return;
    }
    if (input_end > element_count)
    {
        input_end = element_count;
    }

    local_absmax = 0.0f;
    for (element_index = input_begin + threadIdx.x;
         element_index < input_end;
         element_index += blockDim.x)
    {
        float value;

        value = fabsf(SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index]));
        local_absmax = fmaxf(local_absmax, value);
    }

    block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
        local_absmax,
        shared_reduction);
    scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
    scale_block_count = (element_count + scale_block_size - 1u) /
        scale_block_size;

    if (threadIdx.x == 0u)
    {
        uint64_t scale_index;

        scale_index = ((uint64_t)row_index * (uint64_t)scale_block_count) +
            (uint64_t)scale_block_index;
        output_scale_f32[scale_index] = scale_value;
        if (output_amax_f32 != 0)
        {
            output_amax_f32[scale_index] = block_absmax;
        }
    }
    __syncthreads();

    for (element_index = input_begin + threadIdx.x;
         element_index < input_end;
         element_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index]) / scale_value;
        output_fp8_e4m3[((uint64_t)row_index * (uint64_t)element_count) +
            (uint64_t)element_index] =
            SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(input_value);
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageFp8E4m3DequantizeRowsKernel(
    const uint8_t *__restrict__ input_fp8_e4m3,
    const float *__restrict__ input_scale_f32,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t scale_block_size)
{
    uint32_t row_index;
    uint32_t scale_block_index;
    uint32_t input_begin;
    uint32_t input_end;
    uint32_t element_index;
    uint32_t scale_block_count;
    float scale_value;

    row_index = blockIdx.x;
    scale_block_index = blockIdx.y;
    if (row_index >= row_count || scale_block_size == 0u)
    {
        return;
    }

    input_begin = scale_block_index * scale_block_size;
    input_end = input_begin + scale_block_size;
    if (input_begin >= element_count)
    {
        return;
    }
    if (input_end > element_count)
    {
        input_end = element_count;
    }
    scale_block_count = (element_count + scale_block_size - 1u) /
        scale_block_size;
    scale_value = input_scale_f32[
        ((uint64_t)row_index * (uint64_t)scale_block_count) +
        (uint64_t)scale_block_index];

    for (element_index = input_begin + threadIdx.x;
         element_index < input_end;
         element_index += blockDim.x)
    {
        float value;

        value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            input_fp8_e4m3[
                ((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index]) * scale_value;
        output_bf16[((uint64_t)row_index * (uint64_t)element_count) +
            (uint64_t)element_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(value);
    }
}


static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageDivideRoundUpU32(
    uint32_t value,
    uint32_t divisor)
{
    return divisor == 0u ? 0u : (value + divisor - 1u) / divisor;
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
    const uint8_t *__restrict__ payload_fp8_e4m3,
    const float *__restrict__ scale_f32,
    uint64_t row_index,
    uint32_t row_element_count,
    uint32_t element_index,
    uint32_t scale_block_size)
{
    uint32_t scale_block_count;
    uint64_t payload_index;
    uint64_t scale_index;

    scale_block_count = SparkGlm52ResidentDecodeStageDivideRoundUpU32(
        row_element_count,
        scale_block_size);
    payload_index =
        (row_index * (uint64_t)row_element_count) +
        (uint64_t)element_index;
    scale_index =
        (row_index * (uint64_t)scale_block_count) +
        (uint64_t)(element_index / scale_block_size);
    return SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
        payload_fp8_e4m3[payload_index]) * scale_f32[scale_index];
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageFp8E4m3MappedQuantizeRowsKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint32_t *__restrict__ slot_mapping,
    uint8_t *__restrict__ output_fp8_e4m3,
    float *__restrict__ output_scale_f32,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t element_count,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[256u];
    uint32_t sequence_index;
    uint32_t cache_slot_index;
    uint32_t scale_block_index;
    uint32_t input_begin;
    uint32_t input_end;
    uint32_t element_index;
    uint32_t scale_block_count;
    float local_absmax;
    float block_absmax;
    float scale_value;

    sequence_index = blockIdx.x;
    scale_block_index = blockIdx.y;
    if (sequence_index >= active_sequence_count || scale_block_size == 0u)
    {
        return;
    }
    cache_slot_index = slot_mapping[sequence_index];
    if (cache_slot_index >= cache_token_capacity)
    {
        return;
    }

    input_begin = scale_block_index * scale_block_size;
    input_end = input_begin + scale_block_size;
    if (input_begin >= element_count)
    {
        return;
    }
    if (input_end > element_count)
    {
        input_end = element_count;
    }

    local_absmax = 0.0f;
    for (element_index = input_begin + threadIdx.x;
         element_index < input_end;
         element_index += blockDim.x)
    {
        float value;

        value = fabsf(SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)cache_slot_index * (uint64_t)element_count) +
                (uint64_t)element_index]));
        local_absmax = fmaxf(local_absmax, value);
    }

    block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
        local_absmax,
        shared_reduction);
    scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
    scale_block_count = (element_count + scale_block_size - 1u) /
        scale_block_size;

    if (threadIdx.x == 0u)
    {
        output_scale_f32[
            ((uint64_t)cache_slot_index * (uint64_t)scale_block_count) +
            (uint64_t)scale_block_index] = scale_value;
    }
    __syncthreads();

    for (element_index = input_begin + threadIdx.x;
         element_index < input_end;
         element_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)cache_slot_index * (uint64_t)element_count) +
                (uint64_t)element_index]) / scale_value;
        output_fp8_e4m3[
            ((uint64_t)cache_slot_index * (uint64_t)element_count) +
            (uint64_t)element_index] =
            SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(input_value);
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageRmsNormFp8E4m3QuantizeKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint16_t *__restrict__ weight_bf16,
    uint16_t *__restrict__ output_bf16,
    uint8_t *__restrict__ output_fp8_e4m3,
    float *__restrict__ output_scale_f32,
    float *__restrict__ output_amax_f32,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    float epsilon)
{
    __shared__ float shared_reduction[256u];
    __shared__ float shared_inverse_root_mean_square;
    uint32_t row_index;
    uint32_t element_index;
    uint32_t scale_block_index;
    uint32_t scale_block_count;
    float local_square_sum;
    float row_square_sum;

    row_index = blockIdx.x;
    if (row_index >= row_count || scale_block_size == 0u)
    {
        return;
    }

    local_square_sum = 0.0f;
    for (element_index = threadIdx.x;
         element_index < element_count;
         element_index += blockDim.x)
    {
        float value;

        value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_bf16[((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index]);
        local_square_sum += value * value;
    }
    row_square_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_square_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        shared_inverse_root_mean_square = rsqrtf(
            (row_square_sum / (float)element_count) + epsilon);
    }
    __syncthreads();

    scale_block_count = (element_count + scale_block_size - 1u) /
        scale_block_size;
    for (scale_block_index = 0u;
         scale_block_index < scale_block_count;
         ++scale_block_index)
    {
        uint32_t block_begin;
        uint32_t block_end;
        float local_absmax;
        float block_absmax;
        float scale_value;

        block_begin = scale_block_index * scale_block_size;
        block_end = block_begin + scale_block_size;
        if (block_end > element_count)
        {
            block_end = element_count;
        }

        local_absmax = 0.0f;
        for (element_index = block_begin + threadIdx.x;
             element_index < block_end;
             element_index += blockDim.x)
        {
            float value;
            float weight;
            float normalized_value;

            value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[((uint64_t)row_index * (uint64_t)element_count) +
                    (uint64_t)element_index]);
            weight = SparkGlm52ResidentDecodeStageBf16ToFloat(
                weight_bf16[element_index]);
            normalized_value = value * shared_inverse_root_mean_square * weight;
            local_absmax = fmaxf(local_absmax, fabsf(normalized_value));
        }
        block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
            local_absmax,
            shared_reduction);
        scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
        if (threadIdx.x == 0u)
        {
            uint64_t scale_index;

            scale_index =
                ((uint64_t)row_index * (uint64_t)scale_block_count) +
                (uint64_t)scale_block_index;
            output_scale_f32[scale_index] = scale_value;
            if (output_amax_f32 != 0)
            {
                output_amax_f32[scale_index] = block_absmax;
            }
        }
        __syncthreads();

        for (element_index = block_begin + threadIdx.x;
             element_index < block_end;
             element_index += blockDim.x)
        {
            uint64_t output_index;
            float value;
            float weight;
            float normalized_value;

            output_index =
                ((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index;
            value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[output_index]);
            weight = SparkGlm52ResidentDecodeStageBf16ToFloat(
                weight_bf16[element_index]);
            normalized_value = value * shared_inverse_root_mean_square * weight;
            if (output_bf16 != 0)
            {
                output_bf16[output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(normalized_value);
            }
            output_fp8_e4m3[output_index] =
                SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(
                    normalized_value / scale_value);
        }
        __syncthreads();
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52ResidentDecodeStageSiluMulFp8E4m3QuantizeKernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    uint16_t *__restrict__ output_bf16,
    uint8_t *__restrict__ output_fp8_e4m3,
    float *__restrict__ output_scale_f32,
    float *__restrict__ output_amax_f32,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t input_row_stride,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[256u];
    uint32_t row_index;
    uint32_t element_index;
    uint32_t scale_block_index;
    uint32_t scale_block_count;

    row_index = blockIdx.x;
    if (row_index >= row_count || scale_block_size == 0u)
    {
        return;
    }

    scale_block_count = (element_count + scale_block_size - 1u) /
        scale_block_size;
    for (scale_block_index = 0u;
         scale_block_index < scale_block_count;
         ++scale_block_index)
    {
        uint32_t block_begin;
        uint32_t block_end;
        float local_absmax;
        float block_absmax;
        float scale_value;

        block_begin = scale_block_index * scale_block_size;
        block_end = block_begin + scale_block_size;
        if (block_end > element_count)
        {
            block_end = element_count;
        }

        local_absmax = 0.0f;
        for (element_index = block_begin + threadIdx.x;
             element_index < block_end;
             element_index += blockDim.x)
        {
            uint64_t input_index;
            float gate_value;
            float up_value;
            float silu_value;
            float output_value;

            input_index =
                ((uint64_t)row_index * (uint64_t)input_row_stride) +
                (uint64_t)element_index;
            gate_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                gate_bf16[input_index]);
            up_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                up_bf16[input_index]);
            silu_value = gate_value / (1.0f + __expf(-gate_value));
            output_value = silu_value * up_value;
            local_absmax = fmaxf(local_absmax, fabsf(output_value));
        }
        block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
            local_absmax,
            shared_reduction);
        scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
        if (threadIdx.x == 0u)
        {
            uint64_t scale_index;

            scale_index =
                ((uint64_t)row_index * (uint64_t)scale_block_count) +
                (uint64_t)scale_block_index;
            output_scale_f32[scale_index] = scale_value;
            if (output_amax_f32 != 0)
            {
                output_amax_f32[scale_index] = block_absmax;
            }
        }
        __syncthreads();

        for (element_index = block_begin + threadIdx.x;
             element_index < block_end;
             element_index += blockDim.x)
        {
            uint64_t input_index;
            uint64_t output_index;
            float gate_value;
            float up_value;
            float silu_value;
            float output_value;

            input_index =
                ((uint64_t)row_index * (uint64_t)input_row_stride) +
                (uint64_t)element_index;
            output_index =
                ((uint64_t)row_index * (uint64_t)element_count) +
                (uint64_t)element_index;
            gate_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                gate_bf16[input_index]);
            up_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                up_bf16[input_index]);
            silu_value = gate_value / (1.0f + __expf(-gate_value));
            output_value = silu_value * up_value;
            if (output_bf16 != 0)
            {
                output_bf16[output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(output_value);
            }
            output_fp8_e4m3[output_index] =
                SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(
                    output_value / scale_value);
        }
        __syncthreads();
    }
}


static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageWarpDotProductFp8Kv(
    const float *shared_query,
    const uint8_t *key_nope_cache_fp8_e4m3,
    const float *key_nope_cache_scale_f32,
    const uint8_t *mla_cache_fp8_e4m3,
    const float *mla_cache_scale_f32,
    uint32_t cache_slot_index,
    uint32_t head_index,
    uint32_t lane_index,
    uint32_t scale_block_size,
    float qk_scale)
{
    float accumulated_dot_product;
    uint32_t pair_index;

    accumulated_dot_product = 0.0f;
    for (pair_index = lane_index;
         pair_index <
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION / 2u;
         pair_index += SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
    {
        uint32_t first_dimension;
        uint32_t first_key_element_index;

        first_dimension = pair_index * 2u;
        first_key_element_index =
            (head_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
            first_dimension;
        accumulated_dot_product +=
            shared_query[first_dimension] *
            SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                key_nope_cache_fp8_e4m3,
                key_nope_cache_scale_f32,
                cache_slot_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION,
                first_key_element_index,
                scale_block_size);
        accumulated_dot_product +=
            shared_query[first_dimension + 1u] *
            SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                key_nope_cache_fp8_e4m3,
                key_nope_cache_scale_f32,
                cache_slot_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION,
                first_key_element_index + 1u,
                scale_block_size);
    }
    for (pair_index = lane_index;
         pair_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
         pair_index += SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
    {
        uint32_t first_dimension;
        uint32_t first_mla_element_index;

        first_dimension =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
            (pair_index * 2u);
        first_mla_element_index =
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
            (pair_index * 2u);
        accumulated_dot_product +=
            shared_query[first_dimension] *
            SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                mla_cache_fp8_e4m3,
                mla_cache_scale_f32,
                cache_slot_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
                first_mla_element_index,
                scale_block_size);
        accumulated_dot_product +=
            shared_query[first_dimension + 1u] *
            SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                mla_cache_fp8_e4m3,
                mla_cache_scale_f32,
                cache_slot_index,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
                first_mla_element_index + 1u,
                scale_block_size);
    }
    return SparkGlm52ResidentDecodeStageWarpReduceSum(
        accumulated_dot_product) * qk_scale;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageAttentionFp8KvKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint8_t *__restrict__ mla_cache_fp8_e4m3,
    const float *__restrict__ mla_cache_scale_f32,
    const uint8_t *__restrict__ key_nope_cache_fp8_e4m3,
    const float *__restrict__ key_nope_cache_scale_f32,
    const uint8_t *__restrict__ value_cache_fp8_e4m3,
    const float *__restrict__ value_cache_scale_f32,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint16_t *__restrict__ output_value_bf16,
    uint32_t block_token_count,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    uint32_t fp8_scale_block_size,
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
            ? SparkGlm52ResidentDecodeStageWarpDotProductFp8Kv(
                  shared_query,
                  key_nope_cache_fp8_e4m3,
                  key_nope_cache_scale_f32,
                  mla_cache_fp8_e4m3,
                  mla_cache_scale_f32,
                  cache_slot_index,
                  head_index,
                  lane_index,
                  fp8_scale_block_size,
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
    if (row_exponential_sum > 0.0f)
    {
        for (candidate_index = threadIdx.x;
             candidate_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
             candidate_index += blockDim.x)
        {
            shared_scores[candidate_index] /= row_exponential_sum;
        }
    }
    __syncthreads();

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
                    uint32_t cache_element_index;

                    cache_element_index =
                        (head_index *
                         SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                        dimension_index;
                    accumulated_value +=
                        shared_scores[candidate_index] *
                        SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                            value_cache_fp8_e4m3,
                            value_cache_scale_f32,
                            cache_slot_index,
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION,
                            cache_element_index,
                            fp8_scale_block_size);
                }
            }
        }
        output_value_bf16[
            output_row_offset + (uint64_t)dimension_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
    }
}


static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8RowQuantArguments(
    const void *input,
    const void *output,
    const float *scale,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    if (input == 0 || output == 0 || scale == 0 ||
        row_count == 0u || element_count == 0u ||
        scale_block_size == 0u || scale_block_size > 1024u ||
        (scale_block_size & 15u) != 0u ||
        cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize(
    const void *input_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    SparkStatus status;
    dim3 grid;
    cudaError_t cuda_status;

    status = SparkGlm52ResidentDecodeStageValidateFp8RowQuantArguments(
        input_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    grid = dim3(
        active_sequence_count,
        (input_dimension + scale_block_size - 1u) / scale_block_size,
        1u);
    SparkGlm52ResidentDecodeStageFp8E4m3QuantizeRowsKernel<<<
        grid,
        256u,
        0u,
        (cudaStream_t)cuda_stream>>>(
        (const uint16_t *)input_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        output_amax_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3KvCacheStore(
    const void *input_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    uint32_t token_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    return SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize(
        input_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        0,
        token_count,
        element_count,
        scale_block_size,
        cuda_stream);
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3KvCacheLoad(
    const uint8_t *input_fp8_e4m3,
    const float *input_scale_f32,
    void *output_bf16,
    uint32_t token_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    SparkStatus status;
    dim3 grid;
    cudaError_t cuda_status;

    status = SparkGlm52ResidentDecodeStageValidateFp8RowQuantArguments(
        input_fp8_e4m3,
        output_bf16,
        input_scale_f32,
        token_count,
        element_count,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    grid = dim3(
        token_count,
        (element_count + scale_block_size - 1u) / scale_block_size,
        1u);
    SparkGlm52ResidentDecodeStageFp8E4m3DequantizeRowsKernel<<<
        grid,
        256u,
        0u,
        (cudaStream_t)cuda_stream>>>(
        input_fp8_e4m3,
        input_scale_f32,
        (uint16_t *)output_bf16,
        token_count,
        element_count,
        scale_block_size);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MappedRowQuantArguments(
    const void *input,
    const uint32_t *slot_mapping,
    const void *output,
    const float *scale,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    if (input == 0 || slot_mapping == 0 || output == 0 || scale == 0 ||
        active_sequence_count == 0u || cache_token_capacity == 0u ||
        element_count == 0u || scale_block_size == 0u ||
        scale_block_size > 1024u || (scale_block_size & 15u) != 0u ||
        ((element_count + scale_block_size - 1u) / scale_block_size) >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_FUSED_MAX_SCALE_BLOCKS ||
        cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedKvCacheStore(
    const void *input_bf16_cache,
    const uint32_t *slot_mapping,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    uint32_t active_sequence_count,
    uint32_t cache_token_capacity,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    SparkStatus status;
    dim3 grid;
    cudaError_t cuda_status;

    status = SparkGlm52ResidentDecodeStageValidateFp8MappedRowQuantArguments(
        input_bf16_cache,
        slot_mapping,
        output_fp8_e4m3,
        output_scale_f32,
        active_sequence_count,
        cache_token_capacity,
        element_count,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    grid = dim3(
        active_sequence_count,
        (element_count + scale_block_size - 1u) / scale_block_size,
        1u);
    SparkGlm52ResidentDecodeStageFp8E4m3MappedQuantizeRowsKernel<<<
        grid,
        256u,
        0u,
        (cudaStream_t)cuda_stream>>>(
        (const uint16_t *)input_bf16_cache,
        slot_mapping,
        output_fp8_e4m3,
        output_scale_f32,
        active_sequence_count,
        cache_token_capacity,
        element_count,
        scale_block_size);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8FusedQuantArguments(
    const void *first_input,
    const void *second_input,
    const void *output_fp8,
    const float *scale,
    uint32_t row_count,
    uint32_t element_count,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    if (first_input == 0 || second_input == 0 || output_fp8 == 0 ||
        scale == 0 || row_count == 0u || element_count == 0u ||
        scale_block_size == 0u || scale_block_size > 1024u ||
        (scale_block_size & 15u) != 0u ||
        ((element_count + scale_block_size - 1u) / scale_block_size) >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_FUSED_MAX_SCALE_BLOCKS ||
        cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchRmsNormFp8E4m3ActivationQuantize(
    const void *input_bf16,
    const void *weight_bf16,
    void *output_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    float epsilon,
    void *cuda_stream)
{
    SparkStatus status;
    cudaError_t cuda_status;

    status = SparkGlm52ResidentDecodeStageValidateFp8FusedQuantArguments(
        input_bf16,
        weight_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK || !(epsilon > 0.0f))
    {
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkGlm52ResidentDecodeStageRmsNormFp8E4m3QuantizeKernel<<<
        active_sequence_count,
        256u,
        0u,
        (cudaStream_t)cuda_stream>>>(
        (const uint16_t *)input_bf16,
        (const uint16_t *)weight_bf16,
        (uint16_t *)output_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        output_amax_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size,
        epsilon);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchSiluMulFp8E4m3ActivationQuantize(
    const void *gate_bf16,
    const void *up_bf16,
    void *output_bf16,
    uint8_t *output_fp8_e4m3,
    float *output_scale_f32,
    float *output_amax_f32,
    uint32_t active_sequence_count,
    uint32_t input_dimension,
    uint32_t scale_block_size,
    void *cuda_stream)
{
    SparkStatus status;
    cudaError_t cuda_status;

    status = SparkGlm52ResidentDecodeStageValidateFp8FusedQuantArguments(
        gate_bf16,
        up_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        active_sequence_count,
        input_dimension,
        scale_block_size,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ResidentDecodeStageSiluMulFp8E4m3QuantizeKernel<<<
        active_sequence_count,
        256u,
        0u,
        (cudaStream_t)cuda_stream>>>(
        (const uint16_t *)gate_bf16,
        (const uint16_t *)up_bf16,
        (uint16_t *)output_bf16,
        output_fp8_e4m3,
        output_scale_f32,
        output_amax_f32,
        active_sequence_count,
        input_dimension,
        input_dimension,
        scale_block_size);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
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




static bool SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
    const void *pointer,
    uintptr_t alignment);

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
        fp8_moe_plan->gate_up_order !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        fp8_moe_plan->weight_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        fp8_moe_plan->scale_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        fp8_moe_plan->quant_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        fp8_moe_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->launch_function == 0 ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        fp8_moe_plan->validated_maximum_latency_ns == 0u ||
        (fp8_moe_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        (fp8_moe_plan->workspace_bytes != 0u &&
         !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
             fp8_moe_plan->workspace,
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageFp8MoeWeightToFloat(
    const uint8_t *__restrict__ weight_fp8_e4m3,
    const float *__restrict__ weight_scale_inv_f32,
    uint32_t expert_index,
    uint32_t row_index,
    uint32_t input_index,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t scale_block_size)
{
    uint32_t input_scale_block_count;
    uint32_t row_scale_block_count;
    uint64_t payload_index;
    uint64_t scale_index;

    input_scale_block_count = (input_dimension + scale_block_size - 1u) /
        scale_block_size;
    row_scale_block_count = (row_count + scale_block_size - 1u) /
        scale_block_size;
    payload_index =
        (((uint64_t)expert_index * (uint64_t)row_count +
          (uint64_t)row_index) * (uint64_t)input_dimension) +
        (uint64_t)input_index;
    scale_index =
        (((uint64_t)expert_index * (uint64_t)row_scale_block_count +
          (uint64_t)(row_index / scale_block_size)) *
         (uint64_t)input_scale_block_count) +
        (uint64_t)(input_index / scale_block_size);
    return SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            weight_fp8_e4m3[payload_index]) *
        weight_scale_inv_f32[scale_index];
}

static bool SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
    uint64_t *workspace_offset_inout,
    uint64_t element_count,
    uint64_t element_size,
    uint64_t alignment)
{
    uint64_t byte_count;
    uint64_t aligned_byte_count;

    if (workspace_offset_inout == 0 || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u ||
        element_size == 0u || element_count > UINT64_MAX / element_size)
    {
        return false;
    }

    byte_count = element_count * element_size;
    aligned_byte_count = SparkGlm52ResidentDecodeStageAlignUpU64(
        byte_count,
        alignment);
    if (aligned_byte_count < byte_count ||
        *workspace_offset_inout > UINT64_MAX - aligned_byte_count)
    {
        return false;
    }

    *workspace_offset_inout += aligned_byte_count;
    return true;
}

static bool SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
    uint8_t *workspace_base,
    uint64_t workspace_bytes,
    uint64_t *workspace_offset_inout,
    uint64_t element_count,
    uint64_t element_size,
    uint64_t alignment,
    void **range_out)
{
    uint64_t byte_count;
    uint64_t aligned_byte_count;
    uint64_t current_offset;

    if (workspace_base == 0 || workspace_offset_inout == 0 ||
        range_out == 0 || alignment == 0u || element_size == 0u ||
        (alignment & (alignment - 1u)) != 0u ||
        element_count > UINT64_MAX / element_size)
    {
        return false;
    }

    byte_count = element_count * element_size;
    aligned_byte_count = SparkGlm52ResidentDecodeStageAlignUpU64(
        byte_count,
        alignment);
    current_offset = *workspace_offset_inout;
    if (aligned_byte_count < byte_count ||
        current_offset > UINT64_MAX - aligned_byte_count ||
        current_offset + aligned_byte_count > workspace_bytes)
    {
        return false;
    }

    *range_out = workspace_base + current_offset;
    *workspace_offset_inout = current_offset + aligned_byte_count;
    return true;
}

static uint64_t SparkGlm52ResidentDecodeStageFp8MoePackedRouteWorkspaceBytesForShape(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count)
{
    uint64_t maximum_route_count;
    uint64_t workspace_offset;
    uint64_t alignment;

    if (maximum_token_count == 0u || top_k == 0u || expert_count == 0u ||
        maximum_token_count > UINT32_MAX / top_k)
    {
        return 0u;
    }

    maximum_route_count = (uint64_t)maximum_token_count * (uint64_t)top_k;
    alignment =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_PACKED_ROUTE_WORKSPACE_ALIGNMENT_BYTES;
    workspace_offset = 0u;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            (uint64_t)expert_count + 1u,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            expert_count,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            maximum_route_count,
            sizeof(float),
            alignment) ||
        !SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_offset,
            1u,
            sizeof(uint32_t),
            alignment))
    {
        return 0u;
    }
    return workspace_offset;
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoePackedRouteWorkspaceBytes(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count)
{
    return SparkGlm52ResidentDecodeStageFp8MoePackedRouteWorkspaceBytesForShape(
        maximum_token_count,
        top_k,
        expert_count);
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageResolveFp8MoePackedRouteWorkspace(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count,
    void *workspace,
    uint64_t workspace_bytes,
    SparkGlm52Sm121RequiredDecodeStageFp8MoePackedRouteView *packed_route_view_out)
{
    uint8_t *workspace_base;
    uint64_t maximum_route_count;
    uint64_t required_workspace_bytes;
    uint64_t workspace_offset;
    uint64_t alignment;
    void *range_pointer;

    if (packed_route_view_out == 0 || workspace == 0 ||
        maximum_token_count == 0u || top_k == 0u || expert_count == 0u ||
        maximum_token_count > UINT32_MAX / top_k)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            workspace,
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_PACKED_ROUTE_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    maximum_route_count = (uint64_t)maximum_token_count * (uint64_t)top_k;
    if (maximum_route_count > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    required_workspace_bytes =
        SparkGlm52ResidentDecodeStageFp8MoePackedRouteWorkspaceBytesForShape(
            maximum_token_count,
            top_k,
            expert_count);
    if (required_workspace_bytes == 0u || workspace_bytes < required_workspace_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    memset(packed_route_view_out, 0, sizeof(*packed_route_view_out));
    packed_route_view_out->abi_version =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_PACKED_ROUTE_VIEW_ABI_VERSION;
    packed_route_view_out->maximum_token_count = maximum_token_count;
    packed_route_view_out->expert_count = expert_count;
    packed_route_view_out->top_k = top_k;
    packed_route_view_out->maximum_route_count = (uint32_t)maximum_route_count;

    workspace_base = (uint8_t *)workspace;
    workspace_offset = 0u;
    alignment =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_PACKED_ROUTE_WORKSPACE_ALIGNMENT_BYTES;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            (uint64_t)expert_count + 1u,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->expert_route_offsets = (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            expert_count,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->expert_route_counts = (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_expert_ids = (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_source_token_indices = (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_source_route_indices = (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            maximum_route_count,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_route_rows_by_token_route =
        (uint32_t *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            maximum_route_count,
            sizeof(float),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_route_weights = (float *)range_pointer;

    if (!SparkGlm52ResidentDecodeStageAssignAlignedWorkspaceRange(
            workspace_base,
            workspace_bytes,
            &workspace_offset,
            1u,
            sizeof(uint32_t),
            alignment,
            &range_pointer))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    packed_route_view_out->packed_route_count = (uint32_t *)range_pointer;
    return SPARK_STATUS_OK;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageFp8MoePackedFinalizeKernel(
    const uint16_t *__restrict__ packed_down_bf16,
    const uint32_t *__restrict__ packed_route_rows_by_token_route,
    const float *__restrict__ topk_weights,
    uint16_t *__restrict__ output_bf16,
    uint32_t active_sequence_count,
    uint32_t top_k,
    uint32_t hidden_dimension)
{
    uint32_t token_index;
    uint32_t hidden_index;
    uint32_t route_index;
    float accumulated_value;

    token_index = blockIdx.y;
    hidden_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (token_index >= active_sequence_count ||
        hidden_index >= hidden_dimension)
    {
        return;
    }
    accumulated_value = 0.0f;
    for (route_index = 0u; route_index < top_k; ++route_index)
    {
        uint32_t packed_row_index;

        packed_row_index = packed_route_rows_by_token_route[
            (token_index * top_k) + route_index];
        accumulated_value +=
            topk_weights[(token_index * top_k) + route_index] *
            SparkGlm52ResidentDecodeStageBf16ToFloat(packed_down_bf16[
                ((uint64_t)packed_row_index * (uint64_t)hidden_dimension) +
                (uint64_t)hidden_index]);
    }
    output_bf16[
        ((uint64_t)token_index * (uint64_t)hidden_dimension) +
        (uint64_t)hidden_index] =
        SparkGlm52ResidentDecodeStageFloatToBf16(accumulated_value);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageFp8MoePackedRouteResetKernel(
    uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ expert_route_counts,
    uint32_t *__restrict__ packed_expert_ids,
    uint32_t *__restrict__ packed_source_token_indices,
    uint32_t *__restrict__ packed_source_route_indices,
    uint32_t *__restrict__ packed_route_rows_by_token_route,
    float *__restrict__ packed_route_weights,
    uint32_t *__restrict__ packed_route_count,
    uint32_t expert_count,
    uint32_t route_count)
{
    uint32_t linear_index;
    uint32_t linear_stride;
    uint32_t reset_limit;

    linear_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    linear_stride = gridDim.x * blockDim.x;
    reset_limit = route_count > (expert_count + 1u)
        ? route_count
        : (expert_count + 1u);
    for (; linear_index < reset_limit; linear_index += linear_stride)
    {
        if (linear_index <= expert_count)
        {
            expert_route_offsets[linear_index] = 0u;
        }
        if (linear_index < expert_count)
        {
            expert_route_counts[linear_index] = 0u;
        }
        if (linear_index < route_count)
        {
            packed_expert_ids[linear_index] = UINT32_MAX;
            packed_source_token_indices[linear_index] = UINT32_MAX;
            packed_source_route_indices[linear_index] = UINT32_MAX;
            packed_route_rows_by_token_route[linear_index] = UINT32_MAX;
            packed_route_weights[linear_index] = 0.0f;
        }
        if (linear_index == 0u)
        {
            *packed_route_count = 0u;
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageFp8MoePackedRouteCountKernel(
    const uint32_t *__restrict__ topk_expert_ids,
    uint32_t *__restrict__ expert_route_counts,
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k)
{
    __shared__ uint32_t shared_counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t expert_index;
    uint32_t route_count;
    uint32_t route_linear_index;
    uint32_t local_count;
    uint32_t reduction_stride;

    expert_index = blockIdx.x;
    if (expert_index >= expert_count)
    {
        return;
    }

    route_count = active_sequence_count * top_k;
    local_count = 0u;
    for (route_linear_index = threadIdx.x;
         route_linear_index < route_count;
         route_linear_index += blockDim.x)
    {
        if (topk_expert_ids[route_linear_index] == expert_index)
        {
            local_count += 1u;
        }
    }

    shared_counts[threadIdx.x] = local_count;
    __syncthreads();
    for (reduction_stride = blockDim.x >> 1u;
         reduction_stride > 0u;
         reduction_stride >>= 1u)
    {
        if (threadIdx.x < reduction_stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + reduction_stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        expert_route_counts[expert_index] = shared_counts[0];
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageFp8MoePackedRoutePrefixKernel(
    const uint32_t *__restrict__ expert_route_counts,
    uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ packed_route_count,
    uint32_t expert_count)
{
    uint32_t expert_index;
    uint32_t running_offset;

    if (threadIdx.x != 0u || blockIdx.x != 0u)
    {
        return;
    }

    running_offset = 0u;
    for (expert_index = 0u; expert_index < expert_count; ++expert_index)
    {
        expert_route_offsets[expert_index] = running_offset;
        running_offset += expert_route_counts[expert_index];
    }
    expert_route_offsets[expert_count] = running_offset;
    *packed_route_count = running_offset;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageFp8MoePackedRouteFillKernel(
    const uint32_t *__restrict__ topk_expert_ids,
    const float *__restrict__ topk_weights,
    const uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ packed_expert_ids,
    uint32_t *__restrict__ packed_source_token_indices,
    uint32_t *__restrict__ packed_source_route_indices,
    uint32_t *__restrict__ packed_route_rows_by_token_route,
    float *__restrict__ packed_route_weights,
    uint32_t active_sequence_count,
    uint32_t expert_count,
    uint32_t top_k)
{
    uint32_t expert_index;
    uint32_t route_count;
    uint32_t route_linear_index;
    uint32_t write_index;

    expert_index = blockIdx.x;
    if (expert_index >= expert_count || threadIdx.x != 0u)
    {
        return;
    }

    route_count = active_sequence_count * top_k;
    write_index = expert_route_offsets[expert_index];
    for (route_linear_index = 0u;
         route_linear_index < route_count;
         ++route_linear_index)
    {
        if (topk_expert_ids[route_linear_index] == expert_index)
        {
            uint32_t token_index;
            uint32_t route_index;

            token_index = route_linear_index / top_k;
            route_index = route_linear_index - (token_index * top_k);
            packed_expert_ids[write_index] = expert_index;
            packed_source_token_indices[write_index] = token_index;
            packed_source_route_indices[write_index] = route_index;
            packed_route_rows_by_token_route[route_linear_index] = write_index;
            packed_route_weights[write_index] = topk_weights[route_linear_index];
            ++write_index;
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 4)
void SparkGlm52ResidentDecodeStageFp8MoePackedRouteIndptrKernel(
    const uint32_t *__restrict__ expert_route_offsets,
    const uint32_t *__restrict__ expert_route_counts,
    int32_t *__restrict__ m_indptr,
    uint32_t expert_count)
{
    uint32_t expert_index;

    expert_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (expert_index < expert_count)
    {
        m_indptr[expert_index] = (int32_t)expert_route_offsets[expert_index];
    }
    else if (expert_index == expert_count)
    {
        m_indptr[expert_count] =
            (int32_t)(expert_route_offsets[expert_count - 1u] +
                expert_route_counts[expert_count - 1u]);
    }
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MoePackedRouteView(
    const SparkGlm52Sm121RequiredDecodeStageFp8MoePackedRouteView *packed_route_view)
{
    if (packed_route_view == 0 ||
        packed_route_view->abi_version !=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_PACKED_ROUTE_VIEW_ABI_VERSION ||
        packed_route_view->maximum_token_count == 0u ||
        packed_route_view->expert_count == 0u ||
        packed_route_view->top_k == 0u ||
        packed_route_view->maximum_route_count == 0u ||
        packed_route_view->maximum_token_count > UINT32_MAX / packed_route_view->top_k ||
        packed_route_view->maximum_route_count !=
            packed_route_view->maximum_token_count * packed_route_view->top_k ||
        packed_route_view->reserved0 != 0u ||
        packed_route_view->reserved1 != 0u ||
        packed_route_view->expert_route_offsets == 0 ||
        packed_route_view->expert_route_counts == 0 ||
        packed_route_view->packed_expert_ids == 0 ||
        packed_route_view->packed_source_token_indices == 0 ||
        packed_route_view->packed_source_route_indices == 0 ||
        packed_route_view->packed_route_rows_by_token_route == 0 ||
        packed_route_view->packed_route_weights == 0 ||
        packed_route_view->packed_route_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedRouteBuild(
    const uint32_t *topk_expert_ids,
    const float *topk_weights,
    SparkGlm52Sm121RequiredDecodeStageFp8MoePackedRouteView *packed_route_view,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    uint32_t route_count;
    uint32_t reset_limit;
    uint32_t reset_block_count;
    SparkStatus status;

    if (topk_expert_ids == 0 || topk_weights == 0 ||
        cuda_stream_pointer == 0 || active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ResidentDecodeStageValidateFp8MoePackedRouteView(
        packed_route_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (active_sequence_count > packed_route_view->maximum_token_count ||
        active_sequence_count > UINT32_MAX / packed_route_view->top_k)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    packed_route_view->active_sequence_count = active_sequence_count;
    route_count = active_sequence_count * packed_route_view->top_k;
    reset_limit = route_count > (packed_route_view->expert_count + 1u)
        ? route_count
        : (packed_route_view->expert_count + 1u);
    reset_block_count = SparkGlm52ResidentDecodeStageElementBlockCount(reset_limit);
    cuda_stream = (cudaStream_t)cuda_stream_pointer;

    SparkGlm52ResidentDecodeStageFp8MoePackedRouteResetKernel<<<
        reset_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        packed_route_view->expert_route_offsets,
        packed_route_view->expert_route_counts,
        packed_route_view->packed_expert_ids,
        packed_route_view->packed_source_token_indices,
        packed_route_view->packed_source_route_indices,
        packed_route_view->packed_route_rows_by_token_route,
        packed_route_view->packed_route_weights,
        packed_route_view->packed_route_count,
        packed_route_view->expert_count,
        route_count);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    SparkGlm52ResidentDecodeStageFp8MoePackedRouteCountKernel<<<
        packed_route_view->expert_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        topk_expert_ids,
        packed_route_view->expert_route_counts,
        active_sequence_count,
        packed_route_view->expert_count,
        packed_route_view->top_k);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    SparkGlm52ResidentDecodeStageFp8MoePackedRoutePrefixKernel<<<
        1u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        packed_route_view->expert_route_counts,
        packed_route_view->expert_route_offsets,
        packed_route_view->packed_route_count,
        packed_route_view->expert_count);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    SparkGlm52ResidentDecodeStageFp8MoePackedRouteFillKernel<<<
        packed_route_view->expert_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        topk_expert_ids,
        topk_weights,
        packed_route_view->expert_route_offsets,
        packed_route_view->packed_expert_ids,
        packed_route_view->packed_source_token_indices,
        packed_route_view->packed_source_route_indices,
        packed_route_view->packed_route_rows_by_token_route,
        packed_route_view->packed_route_weights,
        active_sequence_count,
        packed_route_view->expert_count,
        packed_route_view->top_k);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageFp8MoePackedHiddenQuantizeKernel(
    const uint16_t *__restrict__ hidden_bf16,
    const uint32_t *__restrict__ packed_source_token_indices,
    uint8_t *__restrict__ packed_hidden_fp8_e4m3,
    float *__restrict__ packed_hidden_scale_f32,
    float *__restrict__ packed_hidden_amax_f32,
    uint32_t routed_row_count,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t scale_block_size)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t packed_route_index;
    uint32_t scale_block_index;
    uint32_t source_token_index;
    uint32_t hidden_begin;
    uint32_t hidden_end;
    uint32_t hidden_index;
    uint32_t hidden_scale_block_count;
    float local_absmax;
    float block_absmax;
    float scale_value;

    packed_route_index = blockIdx.x;
    scale_block_index = blockIdx.y;
    if (packed_route_index >= routed_row_count || scale_block_size == 0u)
    {
        return;
    }

    source_token_index = packed_source_token_indices[packed_route_index];
    if (source_token_index >= active_sequence_count)
    {
        return;
    }

    hidden_begin = scale_block_index * scale_block_size;
    hidden_end = hidden_begin + scale_block_size;
    if (hidden_begin >= hidden_dimension)
    {
        return;
    }
    if (hidden_end > hidden_dimension)
    {
        hidden_end = hidden_dimension;
    }

    local_absmax = 0.0f;
    for (hidden_index = hidden_begin + threadIdx.x;
         hidden_index < hidden_end;
         hidden_index += blockDim.x)
    {
        float hidden_value;

        hidden_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            hidden_bf16[
                ((uint64_t)source_token_index * (uint64_t)hidden_dimension) +
                (uint64_t)hidden_index]);
        local_absmax = fmaxf(local_absmax, fabsf(hidden_value));
    }

    block_absmax = SparkGlm52ResidentDecodeStageBlockReduceMax(
        local_absmax,
        shared_reduction);
    scale_value = fmaxf(block_absmax / 448.0f, 1.0e-8f);
    hidden_scale_block_count =
        (hidden_dimension + scale_block_size - 1u) / scale_block_size;

    if (threadIdx.x == 0u)
    {
        uint64_t scale_index;

        scale_index =
            ((uint64_t)packed_route_index *
             (uint64_t)hidden_scale_block_count) +
            (uint64_t)scale_block_index;
        packed_hidden_scale_f32[scale_index] = scale_value;
        if (packed_hidden_amax_f32 != 0)
        {
            packed_hidden_amax_f32[scale_index] = block_absmax;
        }
    }
    __syncthreads();

    for (hidden_index = hidden_begin + threadIdx.x;
         hidden_index < hidden_end;
         hidden_index += blockDim.x)
    {
        float hidden_value;

        hidden_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            hidden_bf16[
                ((uint64_t)source_token_index * (uint64_t)hidden_dimension) +
                (uint64_t)hidden_index]) / scale_value;
        packed_hidden_fp8_e4m3[
            ((uint64_t)packed_route_index * (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index] =
            SparkGlm52ResidentDecodeStageEncodeFp8E4m3Saturate(hidden_value);
    }
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MoePackedHiddenQuantizeArguments(
    const void *hidden_bf16,
    const uint32_t *packed_source_token_indices,
    const uint8_t *packed_hidden_fp8_e4m3,
    const float *packed_hidden_scale_f32,
    const float *packed_hidden_amax_f32,
    uint32_t routed_row_count,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t scale_block_size,
    void *cuda_stream_pointer)
{
    if (hidden_bf16 == 0 || packed_source_token_indices == 0 ||
        packed_hidden_fp8_e4m3 == 0 || packed_hidden_scale_f32 == 0 ||
        routed_row_count == 0u || active_sequence_count == 0u ||
        hidden_dimension == 0u || scale_block_size == 0u ||
        scale_block_size > 1024u || (scale_block_size & 15u) != 0u ||
        cuda_stream_pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            packed_hidden_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            packed_hidden_scale_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        (packed_hidden_amax_f32 != 0 &&
         !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
             packed_hidden_amax_f32,
             SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedHiddenQuantize(
    const void *hidden_bf16,
    const uint32_t *packed_source_token_indices,
    uint8_t *packed_hidden_fp8_e4m3,
    float *packed_hidden_scale_f32,
    float *packed_hidden_amax_f32,
    uint32_t routed_row_count,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t scale_block_size,
    void *cuda_stream_pointer)
{
    SparkStatus status;
    cudaError_t cuda_status;
    dim3 grid;

    status = SparkGlm52ResidentDecodeStageValidateFp8MoePackedHiddenQuantizeArguments(
        hidden_bf16,
        packed_source_token_indices,
        packed_hidden_fp8_e4m3,
        packed_hidden_scale_f32,
        packed_hidden_amax_f32,
        routed_row_count,
        active_sequence_count,
        hidden_dimension,
        scale_block_size,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    grid = dim3(
        routed_row_count,
        (hidden_dimension + scale_block_size - 1u) / scale_block_size,
        1u);
    SparkGlm52ResidentDecodeStageFp8MoePackedHiddenQuantizeKernel<<<
        grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        (cudaStream_t)cuda_stream_pointer>>>(
        (const uint16_t *)hidden_bf16,
        packed_source_token_indices,
        packed_hidden_fp8_e4m3,
        packed_hidden_scale_f32,
        packed_hidden_amax_f32,
        routed_row_count,
        active_sequence_count,
        hidden_dimension,
        scale_block_size);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

typedef struct SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout
{
    uint64_t route_count;
    uint64_t hidden_scale_block_count;
    uint64_t route_scale_block_count;
    uint64_t hidden_fp8_offset;
    uint64_t hidden_fp8_bytes;
    uint64_t hidden_scale_offset;
    uint64_t hidden_scale_bytes;
    uint64_t hidden_amax_offset;
    uint64_t hidden_amax_bytes;
    uint64_t w1_output_bf16_offset;
    uint64_t w1_output_bf16_bytes;
    uint64_t packed_down_bf16_offset;
    uint64_t packed_down_bf16_bytes;
    uint64_t intermediate_arena_offset;
    uint64_t intermediate_arena_bytes;
    uint64_t intermediate_fp8_offset;
    uint64_t intermediate_fp8_bytes;
    uint64_t intermediate_scale_offset;
    uint64_t intermediate_scale_bytes;
    uint64_t intermediate_amax_offset;
    uint64_t intermediate_amax_bytes;
    uint64_t packed_route_workspace_offset;
    uint64_t packed_route_workspace_bytes;
    uint64_t cutlass_m_indptr_offset;
    uint64_t cutlass_m_indptr_bytes;
    uint64_t cutlass_int_workspace_offset;
    uint64_t cutlass_int_workspace_bytes;
    uint64_t cutlass_float_workspace_offset;
    uint64_t cutlass_float_workspace_bytes;
    uint64_t total_workspace_bytes;
} SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout;

typedef struct SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView
{
    uint8_t *hidden_fp8_e4m3;
    float *hidden_scale_f32;
    float *hidden_amax_f32;
    uint16_t *w1_output_bf16;
    uint16_t *packed_down_bf16;
    uint8_t *intermediate_fp8_e4m3;
    float *intermediate_scale_f32;
    float *intermediate_amax_f32;
    int32_t *cutlass_m_indptr;
    void *cutlass_int_workspace;
    uint64_t cutlass_int_workspace_bytes;
    void *cutlass_float_workspace;
    uint64_t cutlass_float_workspace_bytes;
    SparkGlm52Sm121RequiredDecodeStageFp8MoePackedRouteView packed_route_view;
} SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView;

static bool SparkGlm52ResidentDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceLayout(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension,
    uint32_t scale_block_size,
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout *layout_out)
{
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout layout;
    uint64_t alignment;
    uint64_t route_count;
    uint64_t route_activation_element_count;
    uint64_t route_scale_block_count;
    uint64_t hidden_activation_element_count;
    uint64_t hidden_scale_block_count;
    uint64_t hidden_scale_element_count;
    uint64_t hidden_fp8_bytes;
    uint64_t hidden_scale_bytes;
    uint64_t bf16_route_activation_bytes;
    uint64_t bf16_hidden_route_bytes;
    uint64_t intermediate_fp8_bytes;
    uint64_t intermediate_scale_bytes;
    uint64_t packed_route_workspace_bytes;
    uint64_t workspace_cursor;
    uint64_t arena_cursor;

    if (layout_out == 0)
    {
        return false;
    }
    memset(&layout, 0, sizeof(layout));
    alignment = SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES;

    if (maximum_token_count == 0u || top_k == 0u || expert_count == 0u ||
        hidden_dimension == 0u || intermediate_dimension == 0u ||
        scale_block_size == 0u ||
        maximum_token_count > UINT64_MAX / (uint64_t)top_k)
    {
        return false;
    }

    route_count = (uint64_t)maximum_token_count * (uint64_t)top_k;
    if (route_count == 0u || route_count > UINT32_MAX ||
        route_count > UINT64_MAX / (uint64_t)intermediate_dimension)
    {
        return false;
    }

    if (route_count > UINT64_MAX / (uint64_t)hidden_dimension)
    {
        return false;
    }
    hidden_activation_element_count =
        route_count * (uint64_t)hidden_dimension;
    hidden_fp8_bytes = hidden_activation_element_count;
    hidden_scale_block_count =
        ((uint64_t)hidden_dimension + (uint64_t)scale_block_size - 1u) /
        (uint64_t)scale_block_size;
    if (hidden_scale_block_count == 0u ||
        route_count > UINT64_MAX / hidden_scale_block_count)
    {
        return false;
    }
    hidden_scale_element_count = route_count * hidden_scale_block_count;
    if (hidden_scale_element_count > UINT64_MAX / (uint64_t)sizeof(float))
    {
        return false;
    }
    hidden_scale_bytes = hidden_scale_element_count * (uint64_t)sizeof(float);

    route_activation_element_count =
        route_count * (uint64_t)intermediate_dimension;
    if (route_activation_element_count > UINT64_MAX / (uint64_t)sizeof(uint16_t))
    {
        return false;
    }
    bf16_route_activation_bytes =
        route_activation_element_count * (uint64_t)sizeof(uint16_t);
    if (hidden_activation_element_count > UINT64_MAX / (uint64_t)sizeof(uint16_t))
    {
        return false;
    }
    bf16_hidden_route_bytes =
        hidden_activation_element_count * (uint64_t)sizeof(uint16_t);
    intermediate_fp8_bytes = route_activation_element_count;

    route_scale_block_count =
        ((uint64_t)intermediate_dimension + (uint64_t)scale_block_size - 1u) /
        (uint64_t)scale_block_size;
    if (route_scale_block_count == 0u ||
        route_count > UINT64_MAX / route_scale_block_count ||
        route_count * route_scale_block_count > UINT64_MAX / (uint64_t)sizeof(float))
    {
        return false;
    }
    intermediate_scale_bytes =
        route_count * route_scale_block_count * (uint64_t)sizeof(float);
    packed_route_workspace_bytes =
        SparkGlm52ResidentDecodeStageFp8MoePackedRouteWorkspaceBytesForShape(
            maximum_token_count,
            top_k,
            expert_count);
    if (packed_route_workspace_bytes == 0u)
    {
        return false;
    }

    workspace_cursor = 0u;
    layout.hidden_fp8_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            hidden_fp8_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.hidden_fp8_bytes = workspace_cursor - layout.hidden_fp8_offset;
    layout.hidden_scale_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            hidden_scale_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.hidden_scale_bytes = workspace_cursor - layout.hidden_scale_offset;
    layout.hidden_amax_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            hidden_scale_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.hidden_amax_bytes = workspace_cursor - layout.hidden_amax_offset;

    if (bf16_route_activation_bytes > UINT64_MAX / 2ull)
    {
        return false;
    }
    layout.w1_output_bf16_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            bf16_route_activation_bytes * 2ull,
            1u,
            alignment))
    {
        return false;
    }
    layout.w1_output_bf16_bytes =
        workspace_cursor - layout.w1_output_bf16_offset;
    layout.packed_down_bf16_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            bf16_hidden_route_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.packed_down_bf16_bytes =
        workspace_cursor - layout.packed_down_bf16_offset;

    arena_cursor = 0u;
    layout.intermediate_fp8_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            intermediate_fp8_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.intermediate_fp8_bytes = arena_cursor - layout.intermediate_fp8_offset;
    layout.intermediate_scale_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            intermediate_scale_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.intermediate_scale_bytes = arena_cursor - layout.intermediate_scale_offset;
    layout.intermediate_amax_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            intermediate_scale_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.intermediate_amax_bytes = arena_cursor - layout.intermediate_amax_offset;
    layout.packed_route_workspace_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            packed_route_workspace_bytes,
            1u,
            alignment))
    {
        return false;
    }
    layout.packed_route_workspace_bytes =
        arena_cursor - layout.packed_route_workspace_offset;
    layout.cutlass_m_indptr_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            ((uint64_t)expert_count + 1u) * (uint64_t)sizeof(int32_t),
            1u,
            alignment))
    {
        return false;
    }
    layout.cutlass_m_indptr_bytes =
        arena_cursor - layout.cutlass_m_indptr_offset;
    layout.cutlass_int_workspace_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_CUTLASS_INT_WORKSPACE_BYTES,
            1u,
            alignment))
    {
        return false;
    }
    layout.cutlass_int_workspace_bytes =
        arena_cursor - layout.cutlass_int_workspace_offset;
    layout.cutlass_float_workspace_offset = arena_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &arena_cursor,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_CUTLASS_FLOAT_WORKSPACE_BYTES,
            1u,
            alignment))
    {
        return false;
    }
    layout.cutlass_float_workspace_bytes =
        arena_cursor - layout.cutlass_float_workspace_offset;

    layout.intermediate_arena_offset = workspace_cursor;
    if (!SparkGlm52ResidentDecodeStageAddAlignedWorkspaceRange(
            &workspace_cursor,
            arena_cursor,
            1u,
            alignment))
    {
        return false;
    }
    layout.intermediate_arena_bytes =
        workspace_cursor - layout.intermediate_arena_offset;

    layout.route_count = route_count;
    layout.route_scale_block_count = route_scale_block_count;
    layout.hidden_scale_block_count = hidden_scale_block_count;
    layout.total_workspace_bytes = workspace_cursor;
    *layout_out = layout;
    return true;
}

static uint64_t SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceBytesForShape(
    uint32_t maximum_token_count,
    uint32_t top_k,
    uint32_t expert_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension,
    uint32_t scale_block_size)
{
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout layout;

    if (!SparkGlm52ResidentDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceLayout(
            maximum_token_count,
            top_k,
            expert_count,
            hidden_dimension,
            intermediate_dimension,
            scale_block_size,
            &layout))
    {
        return 0u;
    }
    return layout.total_workspace_bytes;
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    if (fp8_moe_plan == 0 ||
        fp8_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION)
    {
        return 0u;
    }
    return SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceBytesForShape(
        fp8_moe_plan->maximum_token_count,
        fp8_moe_plan->top_k,
        fp8_moe_plan->expert_count,
        fp8_moe_plan->hidden_dimension,
        fp8_moe_plan->intermediate_dimension,
        fp8_moe_plan->scale_block_size);
}

static SparkStatus SparkGlm52ResidentDecodeStageResolveFp8MoeGroupedReferenceWorkspace(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView *workspace_view_out)
{
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceLayout layout;
    uint8_t *workspace_base;
    uint8_t *intermediate_arena_base;
    SparkStatus status;

    if (fp8_moe_plan == 0 || workspace_view_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(workspace_view_out, 0, sizeof(*workspace_view_out));

    if (!SparkGlm52ResidentDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceLayout(
            fp8_moe_plan->maximum_token_count,
            fp8_moe_plan->top_k,
            fp8_moe_plan->expert_count,
            fp8_moe_plan->hidden_dimension,
            fp8_moe_plan->intermediate_dimension,
            fp8_moe_plan->scale_block_size,
            &layout) ||
        fp8_moe_plan->workspace == 0 ||
        fp8_moe_plan->workspace_bytes < layout.total_workspace_bytes ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    workspace_base = (uint8_t *)fp8_moe_plan->workspace;
    intermediate_arena_base = workspace_base + layout.intermediate_arena_offset;
    workspace_view_out->hidden_fp8_e4m3 =
        workspace_base + layout.hidden_fp8_offset;
    workspace_view_out->hidden_scale_f32 =
        (float *)(workspace_base + layout.hidden_scale_offset);
    workspace_view_out->hidden_amax_f32 =
        (float *)(workspace_base + layout.hidden_amax_offset);
    workspace_view_out->w1_output_bf16 =
        (uint16_t *)(workspace_base + layout.w1_output_bf16_offset);
    workspace_view_out->packed_down_bf16 =
        (uint16_t *)(workspace_base + layout.packed_down_bf16_offset);
    workspace_view_out->intermediate_fp8_e4m3 =
        intermediate_arena_base + layout.intermediate_fp8_offset;
    workspace_view_out->intermediate_scale_f32 =
        (float *)(intermediate_arena_base + layout.intermediate_scale_offset);
    workspace_view_out->intermediate_amax_f32 =
        (float *)(intermediate_arena_base + layout.intermediate_amax_offset);
    workspace_view_out->cutlass_m_indptr =
        (int32_t *)(intermediate_arena_base + layout.cutlass_m_indptr_offset);
    workspace_view_out->cutlass_int_workspace =
        (void *)(intermediate_arena_base + layout.cutlass_int_workspace_offset);
    workspace_view_out->cutlass_int_workspace_bytes =
        layout.cutlass_int_workspace_bytes;
    workspace_view_out->cutlass_float_workspace =
        (void *)(intermediate_arena_base + layout.cutlass_float_workspace_offset);
    workspace_view_out->cutlass_float_workspace_bytes =
        layout.cutlass_float_workspace_bytes;

    status = SparkGlm52Sm121RequiredDecodeStageResolveFp8MoePackedRouteWorkspace(
        fp8_moe_plan->maximum_token_count,
        fp8_moe_plan->top_k,
        fp8_moe_plan->expert_count,
        intermediate_arena_base + layout.packed_route_workspace_offset,
        layout.packed_route_workspace_bytes,
        &workspace_view_out->packed_route_view);
    return status;
}

static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedReferenceArguments(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView workspace_view;
    SparkStatus status;

    if (pipeline_slot == 0 || cuda_stream_pointer == 0 ||
        active_sequence_count == 0u ||
        pipeline_slot->post_attention_normalized_hidden_bf16 == 0 ||
        pipeline_slot->moe_router_logits == 0 ||
        pipeline_slot->moe_topk_expert_ids == 0 ||
        pipeline_slot->moe_topk_weights == 0 ||
        pipeline_slot->moe_route_output_bf16 == 0 ||
        node_context == 0 ||
        node_context->moe_router_score_bias_f32 == 0 ||
        node_context->moe_routed_scaling_factor == 0.0f)
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
        active_sequence_count > fp8_moe_plan->maximum_active_sequence_count ||
        fp8_moe_plan->top_k > UINT32_MAX / active_sequence_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SparkGlm52ResidentDecodeStageResolveFp8MoeGroupedReferenceWorkspace(
        fp8_moe_plan,
        &workspace_view);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchFlashInferFp8GroupGemm(
    const uint8_t *activation_fp8_e4m3,
    const float *activation_scale_f32,
    const uint8_t *weight_fp8_e4m3,
    const float *weight_scale_inv_f32,
    uint16_t *output_bf16,
    int32_t *m_indptr,
    void *int_workspace,
    uint64_t int_workspace_bytes,
    void *float_workspace,
    uint64_t float_workspace_bytes,
    uint32_t maximum_route_count,
    uint32_t output_dimension,
    uint32_t input_dimension,
    uint32_t expert_count,
    cudaStream_t cuda_stream)
{
    cudaError_t cuda_status;

    if (activation_fp8_e4m3 == 0 || activation_scale_f32 == 0 ||
        weight_fp8_e4m3 == 0 || weight_scale_inv_f32 == 0 ||
        output_bf16 == 0 || m_indptr == 0 || int_workspace == 0 ||
        float_workspace == 0 || cuda_stream == 0 || maximum_route_count == 0u ||
        output_dimension == 0u || input_dimension == 0u || expert_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_status =
        flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM120<
            1, 128, 128, true, cutlass::float_e4m3_t, cutlass::bfloat16_t>(
            int_workspace,
            (size_t)int_workspace_bytes,
            float_workspace,
            (size_t)float_workspace_bytes,
            (cutlass::float_e4m3_t *)activation_fp8_e4m3,
            (cutlass::float_e4m3_t *)weight_fp8_e4m3,
            (float *)activation_scale_f32,
            (float *)weight_scale_inv_f32,
            (cutlass::bfloat16_t *)output_bf16,
            (int *)m_indptr,
            (int)maximum_route_count,
            (int)output_dimension,
            (int)input_dimension,
            (int)expert_count,
            cuda_stream);
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}


static SparkStatus SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedBackend(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend)
{
    if (fp8_moe_plan == 0 || backend == 0 ||
        backend->abi_version !=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_ABI_VERSION ||
        backend->launch_function == 0 ||
        backend->cuda_architecture != 121u ||
        backend->scale_block_size != fp8_moe_plan->scale_block_size ||
        backend->expert_count != fp8_moe_plan->expert_count ||
        backend->top_k != fp8_moe_plan->top_k ||
        backend->hidden_dimension != fp8_moe_plan->hidden_dimension ||
        backend->intermediate_dimension != fp8_moe_plan->intermediate_dimension ||
        backend->validated_maximum_latency_ns == 0u ||
        (backend->capability_flags &
         SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_REQUIRED_CAPABILITIES) !=
            SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_REQUIRED_CAPABILITIES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static uint64_t SparkGlm52ResidentDecodeStageFp8MoeGroupedExternalBackendWorkspaceOffset(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    uint64_t reference_workspace_bytes;

    reference_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(
            fp8_moe_plan);
    if (reference_workspace_bytes == 0u)
    {
        return 0u;
    }
    return SparkGlm52ResidentDecodeStageAlignUpU64(
        reference_workspace_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES);
}

extern "C" uint64_t SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedExternalBackendWorkspaceBytes(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend)
{
    uint64_t backend_workspace_offset;
    uint64_t backend_workspace_bytes;

    if (SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedBackend(
            fp8_moe_plan,
            backend) != SPARK_STATUS_OK)
    {
        return 0u;
    }
    backend_workspace_offset =
        SparkGlm52ResidentDecodeStageFp8MoeGroupedExternalBackendWorkspaceOffset(
            fp8_moe_plan);
    if (backend_workspace_offset == 0u)
    {
        return 0u;
    }
    backend_workspace_bytes = SparkGlm52ResidentDecodeStageAlignUpU64(
        backend->required_workspace_bytes,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES);
    if (backend_workspace_bytes < backend->required_workspace_bytes ||
        backend_workspace_offset > UINT64_MAX - backend_workspace_bytes)
    {
        return 0u;
    }
    return backend_workspace_offset + backend_workspace_bytes;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedExternalBackend(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend;
    SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackendArguments arguments;
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView workspace_view;
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    uint64_t backend_workspace_offset;
    uint64_t required_workspace_bytes;
    uint32_t routed_row_count;
    SparkStatus status;

    backend = fp8_moe_plan != 0
        ? (const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *)
            fp8_moe_plan->opaque_state
        : 0;
    status = SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedBackend(
        fp8_moe_plan,
        backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedReferenceArguments(
        fp8_moe_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    required_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedExternalBackendWorkspaceBytes(
            fp8_moe_plan,
            backend);
    backend_workspace_offset =
        SparkGlm52ResidentDecodeStageFp8MoeGroupedExternalBackendWorkspaceOffset(
            fp8_moe_plan);
    if (required_workspace_bytes == 0u || backend_workspace_offset == 0u ||
        fp8_moe_plan->workspace == 0 ||
        fp8_moe_plan->workspace_bytes < required_workspace_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageResolveFp8MoeGroupedReferenceWorkspace(
        fp8_moe_plan,
        &workspace_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_stream = (cudaStream_t)cuda_stream_pointer;
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
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52ResidentDecodeStageMaybeForceBenchmarkExpertCoverage(
        pipeline_slot->moe_topk_expert_ids,
        pipeline_slot->moe_topk_weights,
        active_sequence_count,
        node_context->moe_expert_count,
        node_context->moe_top_k,
        node_context->moe_routed_scaling_factor,
        cuda_stream,
        "fp8_grouped_moe");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedRouteBuild(
        pipeline_slot->moe_topk_expert_ids,
        pipeline_slot->moe_topk_weights,
        &workspace_view.packed_route_view,
        active_sequence_count,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    routed_row_count = active_sequence_count * fp8_moe_plan->top_k;
    SparkGlm52ResidentDecodeStageFp8MoePackedRouteIndptrKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(fp8_moe_plan->expert_count + 1u),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        workspace_view.packed_route_view.expert_route_offsets,
        workspace_view.packed_route_view.expert_route_counts,
        workspace_view.cutlass_m_indptr,
        fp8_moe_plan->expert_count);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedHiddenQuantize(
        pipeline_slot->post_attention_normalized_hidden_bf16,
        workspace_view.packed_route_view.packed_source_token_indices,
        workspace_view.hidden_fp8_e4m3,
        workspace_view.hidden_scale_f32,
        workspace_view.hidden_amax_f32,
        routed_row_count,
        active_sequence_count,
        fp8_moe_plan->hidden_dimension,
        fp8_moe_plan->scale_block_size,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&arguments, 0, sizeof(arguments));
    arguments.abi_version =
        SPARK_GLM52_SM121_REQUIRED_DECODE_STAGE_FP8_MOE_GROUPED_BACKEND_ARGUMENTS_ABI_VERSION;
    arguments.active_sequence_count = active_sequence_count;
    arguments.routed_row_count = routed_row_count;
    arguments.expert_count = fp8_moe_plan->expert_count;
    arguments.top_k = fp8_moe_plan->top_k;
    arguments.hidden_dimension = fp8_moe_plan->hidden_dimension;
    arguments.intermediate_dimension = fp8_moe_plan->intermediate_dimension;
    arguments.scale_block_size = fp8_moe_plan->scale_block_size;
    arguments.output_dtype = fp8_moe_plan->output_dtype;
    arguments.hidden_scale_block_count =
        (fp8_moe_plan->hidden_dimension + fp8_moe_plan->scale_block_size - 1u) /
        fp8_moe_plan->scale_block_size;
    arguments.intermediate_scale_block_count =
        (fp8_moe_plan->intermediate_dimension + fp8_moe_plan->scale_block_size - 1u) /
        fp8_moe_plan->scale_block_size;
    arguments.w1_output_scale_block_count = (uint32_t)(
        (((uint64_t)fp8_moe_plan->intermediate_dimension * 2ull) +
         (uint64_t)fp8_moe_plan->scale_block_size - 1ull) /
        (uint64_t)fp8_moe_plan->scale_block_size);
    arguments.w2_output_scale_block_count = arguments.hidden_scale_block_count;
    arguments.packed_route_view = &workspace_view.packed_route_view;
    arguments.packed_hidden_fp8_e4m3 = workspace_view.hidden_fp8_e4m3;
    arguments.packed_hidden_scale_f32 = workspace_view.hidden_scale_f32;
    arguments.packed_hidden_amax_f32 = workspace_view.hidden_amax_f32;
    arguments.w1_weight_fp8_e4m3 = fp8_moe_plan->w1_weight_fp8_e4m3;
    arguments.w1_scale_inv_f32 = fp8_moe_plan->w1_scale_inv_f32;
    arguments.w2_weight_fp8_e4m3 = fp8_moe_plan->w2_weight_fp8_e4m3;
    arguments.w2_scale_inv_f32 = fp8_moe_plan->w2_scale_inv_f32;
    arguments.output_bf16 = pipeline_slot->moe_route_output_bf16;
    arguments.workspace = (uint8_t *)fp8_moe_plan->workspace + backend_workspace_offset;
    arguments.workspace_bytes = fp8_moe_plan->workspace_bytes - backend_workspace_offset;
    arguments.opaque_state = backend->opaque_state;
    arguments.cuda_stream = cuda_stream_pointer;
    return backend->launch_function(&arguments);
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedExternalBackendPlan(
    SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *backend)
{
    uint64_t required_workspace_bytes;

    if (fp8_moe_plan == 0 ||
        fp8_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION ||
        fp8_moe_plan->maximum_active_sequence_count == 0u ||
        fp8_moe_plan->maximum_token_count == 0u ||
        fp8_moe_plan->expert_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        fp8_moe_plan->top_k != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
        fp8_moe_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        fp8_moe_plan->intermediate_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        fp8_moe_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        fp8_moe_plan->cuda_architecture != 121u ||
        fp8_moe_plan->gate_up_order !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        fp8_moe_plan->weight_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        fp8_moe_plan->scale_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        fp8_moe_plan->quant_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        fp8_moe_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        (fp8_moe_plan->capability_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES) !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES ||
        fp8_moe_plan->validated_maximum_latency_ns == 0u ||
        SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedBackend(
            fp8_moe_plan,
            backend) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedExternalBackendWorkspaceBytes(
            fp8_moe_plan,
            backend);
    if (required_workspace_bytes == 0u ||
        fp8_moe_plan->workspace == 0 ||
        fp8_moe_plan->workspace_bytes < required_workspace_bytes ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fp8_moe_plan->opaque_state = (void *)backend;
    fp8_moe_plan->launch_function =
        (void *)SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedExternalBackend;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedReference(
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    SparkGlm52ResidentDecodeStageFp8MoeGroupedReferenceWorkspaceView workspace_view;
    uint32_t routed_row_count;
    dim3 finalize_grid;
    SparkStatus status;

    status = SparkGlm52ResidentDecodeStageValidateFp8MoeGroupedReferenceArguments(
        fp8_moe_plan,
        node_context,
        pipeline_slot,
        active_sequence_count,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageResolveFp8MoeGroupedReferenceWorkspace(
        fp8_moe_plan,
        &workspace_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_stream = (cudaStream_t)cuda_stream_pointer;
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
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52ResidentDecodeStageMaybeForceBenchmarkExpertCoverage(
        pipeline_slot->moe_topk_expert_ids,
        pipeline_slot->moe_topk_weights,
        active_sequence_count,
        node_context->moe_expert_count,
        node_context->moe_top_k,
        node_context->moe_routed_scaling_factor,
        cuda_stream,
        "fp8_grouped_moe");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedRouteBuild(
        pipeline_slot->moe_topk_expert_ids,
        pipeline_slot->moe_topk_weights,
        &workspace_view.packed_route_view,
        active_sequence_count,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    routed_row_count = active_sequence_count * fp8_moe_plan->top_k;
    SparkGlm52ResidentDecodeStageFp8MoePackedRouteIndptrKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(fp8_moe_plan->expert_count + 1u),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        workspace_view.packed_route_view.expert_route_offsets,
        workspace_view.packed_route_view.expert_route_counts,
        workspace_view.cutlass_m_indptr,
        fp8_moe_plan->expert_count);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoePackedHiddenQuantize(
        pipeline_slot->post_attention_normalized_hidden_bf16,
        workspace_view.packed_route_view.packed_source_token_indices,
        workspace_view.hidden_fp8_e4m3,
        workspace_view.hidden_scale_f32,
        workspace_view.hidden_amax_f32,
        routed_row_count,
        active_sequence_count,
        fp8_moe_plan->hidden_dimension,
        fp8_moe_plan->scale_block_size,
        cuda_stream_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchFlashInferFp8GroupGemm(
        workspace_view.hidden_fp8_e4m3,
        workspace_view.hidden_scale_f32,
        fp8_moe_plan->w1_weight_fp8_e4m3,
        fp8_moe_plan->w1_scale_inv_f32,
        workspace_view.w1_output_bf16,
        workspace_view.cutlass_m_indptr,
        workspace_view.cutlass_int_workspace,
        workspace_view.cutlass_int_workspace_bytes,
        workspace_view.cutlass_float_workspace,
        workspace_view.cutlass_float_workspace_bytes,
        routed_row_count,
        fp8_moe_plan->intermediate_dimension * 2u,
        fp8_moe_plan->hidden_dimension,
        fp8_moe_plan->expert_count,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ResidentDecodeStageSiluMulFp8E4m3QuantizeKernel<<<
        routed_row_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        workspace_view.w1_output_bf16 + fp8_moe_plan->intermediate_dimension,
        workspace_view.w1_output_bf16,
        0,
        workspace_view.intermediate_fp8_e4m3,
        workspace_view.intermediate_scale_f32,
        workspace_view.intermediate_amax_f32,
        routed_row_count,
        fp8_moe_plan->intermediate_dimension,
        fp8_moe_plan->intermediate_dimension * 2u,
        fp8_moe_plan->scale_block_size);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    status = SparkGlm52ResidentDecodeStageLaunchFlashInferFp8GroupGemm(
        workspace_view.intermediate_fp8_e4m3,
        workspace_view.intermediate_scale_f32,
        fp8_moe_plan->w2_weight_fp8_e4m3,
        fp8_moe_plan->w2_scale_inv_f32,
        workspace_view.packed_down_bf16,
        workspace_view.cutlass_m_indptr,
        workspace_view.cutlass_int_workspace,
        workspace_view.cutlass_int_workspace_bytes,
        workspace_view.cutlass_float_workspace,
        workspace_view.cutlass_float_workspace_bytes,
        routed_row_count,
        fp8_moe_plan->hidden_dimension,
        fp8_moe_plan->intermediate_dimension,
        fp8_moe_plan->expert_count,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    finalize_grid = dim3(
        (fp8_moe_plan->hidden_dimension +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        active_sequence_count,
        1u);
    SparkGlm52ResidentDecodeStageFp8MoePackedFinalizeKernel<<<
        finalize_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        workspace_view.packed_down_bf16,
        workspace_view.packed_route_view.packed_route_rows_by_token_route,
        pipeline_slot->moe_topk_weights,
        (uint16_t *)pipeline_slot->moe_route_output_bf16,
        active_sequence_count,
        fp8_moe_plan->top_k,
        fp8_moe_plan->hidden_dimension);
    cuda_status = cudaPeekAtLastError();
    return cuda_status == cudaSuccess
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INTERNAL_ERROR;
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageBindFp8MoeGroupedReferencePlan(
    SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan)
{
    uint64_t required_workspace_bytes;

    if (fp8_moe_plan == 0 ||
        fp8_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_PLAN_ABI_VERSION ||
        fp8_moe_plan->maximum_active_sequence_count == 0u ||
        fp8_moe_plan->maximum_token_count == 0u ||
        fp8_moe_plan->expert_count !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT ||
        fp8_moe_plan->top_k != SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K ||
        fp8_moe_plan->hidden_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        fp8_moe_plan->intermediate_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        fp8_moe_plan->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        fp8_moe_plan->cuda_architecture != 121u ||
        fp8_moe_plan->gate_up_order !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_GATE_UP_ORDER_UP_GATE ||
        fp8_moe_plan->weight_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_LAYOUT_EXPERT_MAJOR_ROW_MAJOR ||
        fp8_moe_plan->scale_layout !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_LAYOUT_EXPERT_MAJOR_ROW_BLOCK_MAJOR ||
        fp8_moe_plan->quant_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_QUANT_MODE_E4M3 ||
        fp8_moe_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_BLOCK_SIZE ||
        fp8_moe_plan->reserved0 != 0u ||
        fp8_moe_plan->reserved1 != 0u ||
        fp8_moe_plan->w1_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w1_scale_inv_f32 == 0 ||
        fp8_moe_plan->w2_weight_fp8_e4m3 == 0 ||
        fp8_moe_plan->w2_scale_inv_f32 == 0 ||
        (fp8_moe_plan->capability_flags &
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES) !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_REQUIRED_CAPABILITIES ||
        fp8_moe_plan->validated_maximum_latency_ns == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_workspace_bytes =
        SparkGlm52Sm121RequiredDecodeStageCalculateFp8MoeGroupedReferenceWorkspaceBytes(
            fp8_moe_plan);
    if (required_workspace_bytes == 0u ||
        fp8_moe_plan->workspace == 0 ||
        fp8_moe_plan->workspace_bytes < required_workspace_bytes ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_weight_fp8_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WEIGHT_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w1_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->w2_scale_inv_f32,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_SCALE_ALIGNMENT_BYTES) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_moe_plan->workspace,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_WORKSPACE_ALIGNMENT_BYTES))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fp8_moe_plan->launch_function =
        (void *)SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedReference;
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

    plan_is_required = true;
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

static uint32_t SparkGlm52ResidentDecodeStageDsaIndexShareSelectedTokenCount(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context != 0 &&
        node_context->dsa_indexshare_selected_token_count != 0u)
    {
        return node_context->dsa_indexshare_selected_token_count;
    }
    return SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
}

static uint32_t *SparkGlm52ResidentDecodeStageDsaIndexShareLayerCache(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t layer_index)
{
    uint64_t layer_stride;

    if (node_context == 0 ||
        node_context->selected_token_indices_by_layer == 0 ||
        node_context->max_active_sequence_count == 0u ||
        node_context->dsa_indexshare_layer_count == 0u ||
        layer_index >= node_context->dsa_indexshare_layer_count)
    {
        return 0;
    }
    if (SparkGlm52ResidentDecodeStageDsaIndexShareSelectedTokenCount(
            node_context) !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT)
    {
        return 0;
    }
    layer_stride =
        (uint64_t)node_context->max_active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    return node_context->selected_token_indices_by_layer +
        (layer_stride * (uint64_t)layer_index);
}

static SparkStatus SparkGlm52ResidentDecodeStageCopyDsaIndexShareIndices(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    const uint32_t *input_indices,
    uint32_t *output_indices,
    uint32_t active_sequence_count)
{
    uint64_t element_count;
    dim3 copy_grid;

    if (node_context == 0 || input_indices == 0 || output_indices == 0 ||
        cuda_stream == 0 || active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
    copy_grid = dim3(
        (uint32_t)((element_count +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS),
        1u,
        1u);
    SparkGlm52ResidentDecodeStageCopySelectedTokenIndicesKernel<<<
        copy_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        input_indices,
        output_indices,
        element_count);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    uint32_t *layer_cache;
    SparkStatus status;

    if (node_context->sparse_index_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED)
    {
        return SPARK_STATUS_OK;
    }
    if (node_context->sparse_index_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED)
    {
        layer_cache = SparkGlm52ResidentDecodeStageDsaIndexShareLayerCache(
            node_context,
            node_context->dsa_indexshare_source_layer_index);
        return SparkGlm52ResidentDecodeStageCopyDsaIndexShareIndices(
            node_context,
            cuda_slot_state,
            cuda_stream,
            layer_cache,
            pipeline_slot->sparse_token_indices,
            active_sequence_count);
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
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
    }
    else
    {
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
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
    }
    if (status != SPARK_STATUS_OK ||
        node_context->sparse_index_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL)
    {
        return status;
    }
    layer_cache = SparkGlm52ResidentDecodeStageDsaIndexShareLayerCache(
        node_context,
        node_context->layer_index);
    return SparkGlm52ResidentDecodeStageCopyDsaIndexShareIndices(
        node_context,
        cuda_slot_state,
        cuda_stream,
        pipeline_slot->sparse_token_indices,
        layer_cache,
        active_sequence_count);
}


static bool SparkGlm52ResidentDecodeStageLinearPlanUsesFp8E4m3QuantizedView(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const SparkGlm52ResidentDecodeStageQuantizedLinearView **view_out)
{
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *view;

    if (view_out != 0)
    {
        *view_out = 0;
    }
    if (linear_plan == 0 ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            linear_plan,
            &view) ||
        view->weight_format !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3 ||
        linear_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
    {
        return false;
    }
    if (view_out != 0)
    {
        *view_out = view;
    }
    return true;
}

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchFp8DenseMlpPreparedStaging(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *gate_plan;
    const SparkGlm52ResidentDecodeStageLinearPlan *up_plan;
    const SparkGlm52ResidentDecodeStageLinearPlan *down_plan;
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *gate_view;
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *up_view;
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *down_view;
    uint8_t *shared_activation_fp8_e4m3;
    float *shared_activation_scale_f32;
    float *shared_activation_amax_f32;
    uint8_t *down_activation_fp8_e4m3;
    float *down_activation_scale_f32;
    float *down_activation_amax_f32;
    uint64_t hidden_element_count;
    SparkStatus status;

    if (node_context == 0 || pipeline_slot == 0 || cuda_stream == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->mlp_execution_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (pipeline_slot->post_attention_normalized_hidden_bf16 == 0 ||
        pipeline_slot->moe_gate_bf16 == 0 || pipeline_slot->moe_up_bf16 == 0 ||
        pipeline_slot->moe_intermediate_bf16 == 0 ||
        pipeline_slot->layer_output_hidden_bf16 == 0 ||
        pipeline_slot->post_attention_hidden_bf16 == 0 ||
        node_context->dense_intermediate_dimension == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    gate_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension,
        active_sequence_count);
    up_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->dense_intermediate_dimension,
        active_sequence_count);
    down_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN,
        node_context->dense_intermediate_dimension,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        active_sequence_count);
    if (!SparkGlm52ResidentDecodeStageLinearPlanUsesFp8E4m3QuantizedView(
            gate_plan,
            &gate_view) ||
        !SparkGlm52ResidentDecodeStageLinearPlanUsesFp8E4m3QuantizedView(
            up_plan,
            &up_view) ||
        !SparkGlm52ResidentDecodeStageLinearPlanUsesFp8E4m3QuantizedView(
            down_plan,
            &down_view))
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    if (gate_plan->output_is_f32 != 0u || up_plan->output_is_f32 != 0u ||
        down_plan->output_is_f32 != 0u ||
        gate_view->scale_block_size != up_view->scale_block_size ||
        gate_view->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK ||
        down_view->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearWorkspace(
        gate_plan->workspace,
        gate_plan->workspace_bytes,
        gate_plan->maximum_active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        gate_view->scale_block_size,
        &shared_activation_fp8_e4m3,
        &shared_activation_scale_f32,
        &shared_activation_amax_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationQuantize(
        pipeline_slot->post_attention_normalized_hidden_bf16,
        shared_activation_fp8_e4m3,
        shared_activation_scale_f32,
        shared_activation_amax_f32,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        gate_view->scale_block_size,
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

    status = SparkGlm52ResidentDecodeStageLaunchFp8PreparedActivationWeightLinearPlan(
        gate_plan,
        gate_view,
        shared_activation_fp8_e4m3,
        shared_activation_scale_f32,
        shared_activation_amax_f32,
        pipeline_slot->moe_gate_bf16,
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

    status = SparkGlm52ResidentDecodeStageLaunchFp8PreparedActivationWeightLinearPlan(
        up_plan,
        up_view,
        shared_activation_fp8_e4m3,
        shared_activation_scale_f32,
        shared_activation_amax_f32,
        pipeline_slot->moe_up_bf16,
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

    status = SparkGlm52ResidentDecodeStageResolveFp8ActivationLinearWorkspace(
        down_plan->workspace,
        down_plan->workspace_bytes,
        down_plan->maximum_active_sequence_count,
        node_context->dense_intermediate_dimension,
        down_view->scale_block_size,
        &down_activation_fp8_e4m3,
        &down_activation_scale_f32,
        &down_activation_amax_f32);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchSiluMulFp8E4m3ActivationQuantize(
        pipeline_slot->moe_gate_bf16,
        pipeline_slot->moe_up_bf16,
        pipeline_slot->moe_intermediate_bf16,
        down_activation_fp8_e4m3,
        down_activation_scale_f32,
        down_activation_amax_f32,
        active_sequence_count,
        node_context->dense_intermediate_dimension,
        down_view->scale_block_size,
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

    status = SparkGlm52ResidentDecodeStageLaunchFp8PreparedActivationWeightLinearPlan(
        down_plan,
        down_view,
        down_activation_fp8_e4m3,
        down_activation_scale_f32,
        down_activation_amax_f32,
        pipeline_slot->layer_output_hidden_bf16,
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

    status = SparkGlm52ResidentDecodeStageTryLaunchFp8DenseMlpPreparedStaging(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count);
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }

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
    const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *fp8_scaled_gemm_backend;
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
            if (quantized_view->weight_format ==
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3 &&
                linear_plan->algorithm != 0)
            {
                fp8_scaled_gemm_backend =
                    (const SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend *)
                    linear_plan->algorithm;
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->abi_version);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->capability_flags);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->cuda_architecture);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->scale_block_size);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->minimum_m_alignment);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->minimum_n_alignment);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->minimum_k_alignment);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    SparkGlm52ResidentDecodeStagePointerGraphSignature(
                        (const void *)fp8_scaled_gemm_backend->launch_function));
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    SparkGlm52ResidentDecodeStagePointerGraphSignature(
                        fp8_scaled_gemm_backend->opaque_state));
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->required_workspace_bytes);
                signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
                    signature,
                    fp8_scaled_gemm_backend->validated_maximum_latency_ns);
            }
        }
    }
    return signature;
}

static uint64_t SparkGlm52ResidentDecodeStageMixFp8MoePlanGraphSignature(
    uint64_t signature,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8MoePlan *fp8_moe_plan;
    const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *fp8_moe_grouped_backend;

    fp8_moe_plan = node_context != 0 ? node_context->fp8_moe_plan : 0;
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(fp8_moe_plan));
    if (fp8_moe_plan == 0)
    {
        return signature;
    }

    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->abi_version);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->capability_flags);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->maximum_active_sequence_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->maximum_token_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->expert_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->top_k);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->hidden_dimension);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->intermediate_dimension);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->output_dtype);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->cuda_architecture);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->gate_up_order);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->weight_layout);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->scale_layout);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->quant_mode);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->scale_block_size);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->launch_function));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->opaque_state));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->w1_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->w1_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->w2_weight_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->w2_scale_inv_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_moe_plan->workspace));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->workspace_bytes);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_moe_plan->validated_maximum_latency_ns);
    if (fp8_moe_plan->launch_function ==
            (void *)SparkGlm52Sm121RequiredDecodeStageLaunchFp8MoeGroupedExternalBackend &&
        fp8_moe_plan->opaque_state != 0)
    {
        fp8_moe_grouped_backend =
            (const SparkGlm52Sm121RequiredDecodeStageFp8MoeGroupedBackend *)
            fp8_moe_plan->opaque_state;
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->abi_version);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->capability_flags);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->cuda_architecture);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->scale_block_size);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->expert_count);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->top_k);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->hidden_dimension);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->intermediate_dimension);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                (const void *)fp8_moe_grouped_backend->launch_function));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            SparkGlm52ResidentDecodeStagePointerGraphSignature(
                fp8_moe_grouped_backend->opaque_state));
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->required_workspace_bytes);
        signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
            signature,
            fp8_moe_grouped_backend->validated_maximum_latency_ns);
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



static bool SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
    const void *pointer,
    uintptr_t alignment)
{
    if (pointer == 0 || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u)
    {
        return false;
    }
    return (((uintptr_t)pointer) & (alignment - 1u)) == 0u;
}

static bool SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    return node_context != 0 &&
        (node_context->reserved_execution_flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE) != 0u;
}

static bool SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;
    uint32_t required_capabilities;

    if (node_context == 0 || node_context->fp8_kv_cache_plan == 0)
    {
        return false;
    }

    fp8_kv_cache_plan = node_context->fp8_kv_cache_plan;
    required_capabilities =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_REQUIRED_CAPABILITIES;
    if (fp8_kv_cache_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_PLAN_ABI_VERSION ||
        fp8_kv_cache_plan->maximum_active_sequence_count <
            node_context->max_active_sequence_count ||
        fp8_kv_cache_plan->cache_token_capacity <
            node_context->cache_token_capacity ||
        fp8_kv_cache_plan->cache_token_elements !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS ||
        fp8_kv_cache_plan->key_nope_elements !=
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) ||
        fp8_kv_cache_plan->value_elements !=
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) ||
        fp8_kv_cache_plan->scale_block_size !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK ||
        fp8_kv_cache_plan->validated_maximum_latency_ns == 0u ||
        (fp8_kv_cache_plan->capability_flags & required_capabilities) !=
            required_capabilities ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->mla_cache_fp8_e4m3,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->mla_cache_scale_f32,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->key_nope_cache_fp8_e4m3,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->key_nope_cache_scale_f32,
            4u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->value_cache_fp8_e4m3,
            1u) ||
        !SparkGlm52ResidentDecodeStagePointerIsAlignedCuda(
            fp8_kv_cache_plan->value_cache_scale_f32,
            4u))
    {
        return false;
    }
    return true;
}

static uint64_t SparkGlm52ResidentDecodeStageMixFp8KvCachePlanGraphSignature(
    uint64_t signature,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;

    fp8_kv_cache_plan = node_context != 0 ? node_context->fp8_kv_cache_plan : 0;
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(fp8_kv_cache_plan));
    if (fp8_kv_cache_plan == 0)
    {
        return signature;
    }

    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->abi_version);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->capability_flags);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->maximum_active_sequence_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->cache_token_capacity);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->cache_token_elements);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->key_nope_elements);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->value_elements);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->scale_block_size);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->mla_cache_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->mla_cache_scale_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->key_nope_cache_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->key_nope_cache_scale_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->value_cache_fp8_e4m3));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->value_cache_scale_f32));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        SparkGlm52ResidentDecodeStagePointerGraphSignature(
            fp8_kv_cache_plan->workspace));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->workspace_bytes);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(
        signature,
        fp8_kv_cache_plan->validated_maximum_latency_ns);
    return signature;
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
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->layer_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->dsa_indexshare_source_layer_index);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->dsa_indexshare_group_end_layer_exclusive);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->dsa_indexshare_selected_token_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->dsa_indexshare_layer_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->selected_token_indices_by_layer));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, node_context->linear_plan_count);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->linear_plans));
    signature = SparkGlm52ResidentDecodeStageMixLinearPlansGraphSignature(signature, node_context);
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->b12x_moe_dispatch_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->restricted_logits_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->mtp_draft_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->full_stage_plan));
    signature = SparkGlm52ResidentDecodeStageMixGraphSignature(signature, SparkGlm52ResidentDecodeStagePointerGraphSignature(node_context->bulk_prefill_plan));
    signature = SparkGlm52ResidentDecodeStageMixFp8MoePlanGraphSignature(signature, node_context);
    signature = SparkGlm52ResidentDecodeStageMixFp8KvCachePlanGraphSignature(signature, node_context);
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
        pipeline_slot->restricted_logits,
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


static SparkStatus SparkGlm52ResidentDecodeStageLaunchFp8KvCacheShadowStore(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;
    SparkStatus status;

    if (!SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
            node_context))
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
            node_context) ||
        pipeline_slot == 0 ||
        pipeline_slot->slot_mapping == 0 ||
        cuda_stream == 0 ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        node_context->mla_cache_bf16 == 0 ||
        node_context->key_nope_cache_bf16 == 0 ||
        node_context->value_cache_bf16 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fp8_kv_cache_plan = node_context->fp8_kv_cache_plan;
    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedKvCacheStore(
        node_context->mla_cache_bf16,
        pipeline_slot->slot_mapping,
        fp8_kv_cache_plan->mla_cache_fp8_e4m3,
        fp8_kv_cache_plan->mla_cache_scale_f32,
        active_sequence_count,
        node_context->cache_token_capacity,
        fp8_kv_cache_plan->cache_token_elements,
        fp8_kv_cache_plan->scale_block_size,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedKvCacheStore(
        node_context->key_nope_cache_bf16,
        pipeline_slot->slot_mapping,
        fp8_kv_cache_plan->key_nope_cache_fp8_e4m3,
        fp8_kv_cache_plan->key_nope_cache_scale_f32,
        active_sequence_count,
        node_context->cache_token_capacity,
        fp8_kv_cache_plan->key_nope_elements,
        fp8_kv_cache_plan->scale_block_size,
        (void *)cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3MappedKvCacheStore(
        node_context->value_cache_bf16,
        pipeline_slot->slot_mapping,
        fp8_kv_cache_plan->value_cache_fp8_e4m3,
        fp8_kv_cache_plan->value_cache_scale_f32,
        active_sequence_count,
        node_context->cache_token_capacity,
        fp8_kv_cache_plan->value_elements,
        fp8_kv_cache_plan->scale_block_size,
        (void *)cuda_stream);
}

static SparkStatus SparkGlm52ResidentDecodeStageLaunchAbsorbedLatentAttention(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    const SparkGlm52ResidentDecodeStageLinearPlan *kv_b_plan;
    const SparkGlm52ResidentDecodeStageQuantizedLinearView *quantized_view;
    dim3 project_grid;
    dim3 commit_grid;
    dim3 attention_grid;
    dim3 value_grid;
    uint64_t commit_element_count;
    SparkStatus status;

    if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
            node_context) ||
        node_context->mla_cache_bf16 == 0 ||
        pipeline_slot->raw_kv_b_bf16 == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    kv_b_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
        node_context,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,
        active_sequence_count);
    if (kv_b_plan == 0 ||
        !SparkGlm52ResidentDecodeStagePlanKindIsBlackwellQuantizedTensorCore(
            kv_b_plan->plan_kind) ||
        !SparkGlm52ResidentDecodeStageQuantizedLinearViewIsUsable(
            kv_b_plan,
            &quantized_view))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    project_grid = dim3(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE,
        1u);
    SparkGlm52ResidentDecodeStageAbsorbedQueryProjectKernel<<<
        project_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->query_latent_bf16,
        (const uint8_t *)quantized_view->weight_payload,
        quantized_view->weight_scale,
        (uint16_t *)pipeline_slot->query_latent_bf16,
        (uint16_t *)pipeline_slot->raw_kv_b_bf16,
        active_sequence_count,
        quantized_view->weight_format,
        quantized_view->scale_block_size);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    commit_element_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION;
    commit_grid = dim3(
        (uint32_t)((commit_element_count +
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS - 1u) /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS),
        1u,
        1u);
    SparkGlm52ResidentDecodeStageAbsorbedQueryCommitKernel<<<
        commit_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->raw_kv_b_bf16,
        (uint16_t *)pipeline_slot->query_latent_bf16,
        active_sequence_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    attention_grid = dim3(
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_HEADS_PER_BLOCK,
        1u);
    SparkGlm52ResidentDecodeStageAbsorbedAttentionKernel<<<
        attention_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_ATTENTION_THREADS,
        0u,
        cuda_stream>>>(
        (uint16_t *)pipeline_slot->query_latent_bf16,
        (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
        (const uint16_t *)node_context->mla_cache_bf16,
        pipeline_slot->block_table,
        pipeline_slot->context_lengths,
        pipeline_slot->first_block_token_offsets,
        pipeline_slot->sparse_token_indices,
        SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context),
        node_context->max_blocks_per_sequence,
        node_context->kv_block_count,
        node_context->cache_token_capacity,
        node_context->qk_scale);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    value_grid = dim3(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION /
            SPARK_GLM52_RESIDENT_DECODE_STAGE_ABSORBED_OUT_TILE,
        1u);
    SparkGlm52ResidentDecodeStageAbsorbedValueApplyKernel<<<
        value_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->query_latent_bf16,
        (const uint8_t *)quantized_view->weight_payload,
        quantized_view->weight_scale,
        (uint16_t *)pipeline_slot->attention_output_latent_bf16,
        active_sequence_count,
        quantized_view->weight_format,
        quantized_view->scale_block_size);
    return SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
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

    if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
            node_context) &&
        (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
            node_context) ||
         node_context->mla_cache_bf16 == 0 ||
         node_context->key_nope_cache_bf16 == 0 ||
         node_context->value_cache_bf16 == 0))
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "fp8_kv_prepare_contract",
            SPARK_STATUS_INVALID_ARGUMENT);
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
    status = SparkGlm52ResidentDecodeStageLaunchFp8KvCacheShadowStore(
        node_context,
        pipeline_slot,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
            "fp8_kv_write",
            status);
    }
    if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
            node_context))
    {
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
                "fp8_kv_write_check",
                status);
        }
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
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_ABSORBED_LATENT)
    {
        status = SparkGlm52ResidentDecodeStageLaunchAbsorbedLatentAttention(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
                "absorbed_latent_attention",
                status);
        }
    }
    else if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
            node_context))
    {
        const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;

        if (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
                node_context))
        {
            return SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(
                "fp8_kv_attention_contract",
                SPARK_STATUS_INVALID_ARGUMENT);
        }
        fp8_kv_cache_plan = node_context->fp8_kv_cache_plan;
        SparkGlm52ResidentDecodeStageAttentionFp8KvKernel<<<
            attention_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->query_latent_bf16,
            (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
            fp8_kv_cache_plan->mla_cache_fp8_e4m3,
            fp8_kv_cache_plan->mla_cache_scale_f32,
            fp8_kv_cache_plan->key_nope_cache_fp8_e4m3,
            fp8_kv_cache_plan->key_nope_cache_scale_f32,
            fp8_kv_cache_plan->value_cache_fp8_e4m3,
            fp8_kv_cache_plan->value_cache_scale_f32,
            pipeline_slot->block_table,
            pipeline_slot->context_lengths,
            pipeline_slot->first_block_token_offsets,
            pipeline_slot->sparse_token_indices,
            (uint16_t *)pipeline_slot->attention_output_latent_bf16,
            SparkGlm52ResidentDecodeStageEffectiveKvBlockTokenCountCuda(node_context),
            node_context->max_blocks_per_sequence,
            node_context->kv_block_count,
            node_context->cache_token_capacity,
            fp8_kv_cache_plan->scale_block_size,
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

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareExportSelectedTokens(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t source_layer_index,
    void *selected_token_sideband,
    uint32_t active_sequence_count,
    void *cuda_stream)
{
    uint32_t *layer_cache;

    layer_cache = SparkGlm52ResidentDecodeStageDsaIndexShareLayerCache(
        node_context,
        source_layer_index);
    return SparkGlm52ResidentDecodeStageCopyDsaIndexShareIndices(
        node_context,
        0,
        (cudaStream_t)cuda_stream,
        layer_cache,
        (uint32_t *)selected_token_sideband,
        active_sequence_count);
}

extern "C" SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunchDsaIndexShareImportSelectedTokens(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const void *selected_token_sideband,
    uint32_t source_layer_index,
    uint32_t active_sequence_count,
    void *cuda_stream)
{
    uint32_t *layer_cache;

    layer_cache = SparkGlm52ResidentDecodeStageDsaIndexShareLayerCache(
        node_context,
        source_layer_index);
    return SparkGlm52ResidentDecodeStageCopyDsaIndexShareIndices(
        node_context,
        0,
        (cudaStream_t)cuda_stream,
        (const uint32_t *)selected_token_sideband,
        layer_cache,
        active_sequence_count);
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
        SparkGlm52StagePlanBatchBucketIsSupported(
            exact_stage_slice_plan->batch_bucket) == 0u ||
        exact_stage_slice_plan->maximum_active_sequence_count < active_sequence_count ||
        exact_stage_slice_plan->validated_maximum_latency_ns == 0u ||
        (exact_stage_slice_plan->capability_flags & exact_required_capabilities) !=
            exact_required_capabilities ||
        layer_count != 6u ||
        (final_token_stage != 0u &&
         exact_stage_slice_plan->first_layer_index + 6u !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT))
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

#define SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(stage_index, final_token_stage) \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B16, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B32, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B64, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B128, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B256, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B512, final_token_stage), \
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B1024, final_token_stage)

static const SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLauncher
    SparkGlm52ResidentDecodeStageBuiltinExactPp13StageSliceLaunchers[] =
{
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(0u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(1u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(2u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(3u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(4u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(5u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(6u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(7u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(8u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(9u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(10u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(11u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 0u),
    SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 1u)
};

#undef SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS
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
            SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_PAGED_ATTENTION) != 0u)
    {
        if (paged_prefill_plan->prompt_query_latent_bf16 == 0 ||
            paged_prefill_plan->prompt_rotated_query_rope_bf16 == 0 ||
            paged_prefill_plan->prompt_attention_output_latent_bf16 == 0 ||
            node_context->cache_token_capacity == 0u ||
            node_context->kv_block_count == 0u ||
            node_context->max_blocks_per_sequence == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
                node_context))
        {
            if (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
                    node_context))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (node_context->mla_cache_bf16 == 0 ||
                 node_context->key_nope_cache_bf16 == 0 ||
                 node_context->value_cache_bf16 == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
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



#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS \
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS
static_assert(
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_QUERY_TILE_TOKENS == 16u &&
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PAGED_PREFILL_KEY_TILE_TOKENS == 16u,
    "wmma prefill kernel requires 16x16 tile geometry");
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_THREADS 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_WARPS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_THREADS / \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_WARP_VALUE_DIMENSIONS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION / \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_WARPS)

template <uint32_t kKvCacheIsFp8>
static __global__ __launch_bounds__(
    SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_THREADS, 2)
void SparkGlm52ResidentDecodeStagePagedPrefillAttentionWmmaKernel(
    const uint16_t *__restrict__ prompt_query_latent_bf16,
    const uint16_t *__restrict__ prompt_rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint16_t *__restrict__ key_nope_cache_bf16,
    const uint16_t *__restrict__ value_cache_bf16,
    const uint8_t *__restrict__ mla_cache_fp8_e4m3,
    const float *__restrict__ mla_cache_scale_f32,
    const uint8_t *__restrict__ key_nope_cache_fp8_e4m3,
    const float *__restrict__ key_nope_cache_scale_f32,
    const uint8_t *__restrict__ value_cache_fp8_e4m3,
    const float *__restrict__ value_cache_scale_f32,
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
    float qk_scale,
    uint32_t fp8_scale_block_size)
{
    __shared__ __align__(32) __nv_bfloat16 shared_query_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION];
    __shared__ __align__(32) __nv_bfloat16 shared_key_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION];
    __shared__ __align__(32) __nv_bfloat16 shared_value_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION];
    __shared__ __align__(32) float shared_output_accumulator[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION];
    __shared__ __align__(32) float shared_score_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ __align__(32) __nv_bfloat16 shared_probability_tile[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS]
        [SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ uint32_t shared_query_positions[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ uint32_t shared_query_context_lengths[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ uint32_t shared_query_valid[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ float shared_row_maximum[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ float shared_row_denominator[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ float shared_row_scale[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS];
    __shared__ uint32_t shared_maximum_context_length;
    uint32_t sequence_index;
    uint32_t query_tile_base;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t first_block_token_offset;
    uint32_t lane_prompt_token_count;
    uint32_t effective_prompt_token_stride;
    uint32_t element_index;
    uint32_t tile_index;
    uint32_t candidate_base;

    sequence_index = blockIdx.x;
    query_tile_base =
        blockIdx.y * SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS;
    head_index = blockIdx.z;
    if (sequence_index >= active_sequence_count ||
        head_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT)
    {
        return;
    }
    lane_index = threadIdx.x & (SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES - 1u);
    warp_index = threadIdx.x / SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES;
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
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS)
    {
        uint32_t prompt_token_index;
        uint32_t current_position;
        uint32_t context_length;

        tile_index = threadIdx.x;
        prompt_token_index = query_tile_base + tile_index;
        if (prompt_token_index < lane_prompt_token_count)
        {
            current_position = prompt_token_offset + prompt_positions[
                ((uint64_t)sequence_index *
                 (uint64_t)effective_prompt_token_stride) +
                (uint64_t)prompt_token_index];
            context_length = prompt_context_lengths[sequence_index];
            if (context_length == 0u || context_length > current_position + 1u)
                context_length = current_position + 1u;
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
        shared_row_maximum[tile_index] = -FLT_MAX;
        shared_row_denominator[tile_index] = 0.0f;
        shared_row_scale[tile_index] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        uint32_t maximum_context_length;

        maximum_context_length = 0u;
        for (tile_index = 0u;
             tile_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS;
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
    for (element_index = threadIdx.x;
         element_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
         element_index += blockDim.x)
    {
        uint32_t query_dimension;
        uint64_t query_row_index;
        uint16_t query_value;

        tile_index =
            element_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
        query_dimension =
            element_index - (tile_index *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION);
        query_value = 0u;
        if (query_tile_base + tile_index < lane_prompt_token_count)
        {
            query_row_index =
                ((((uint64_t)sequence_index *
                   (uint64_t)effective_prompt_token_stride) +
                  (uint64_t)(query_tile_base + tile_index)) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                (uint64_t)head_index;
            query_value = query_dimension <
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION
                ? prompt_query_latent_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                    (uint64_t)query_dimension]
                : prompt_rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                    (uint64_t)(query_dimension -
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)];
        }
        ((uint16_t *)shared_query_tile)[element_index] = query_value;
    }
    for (element_index = threadIdx.x;
         element_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
         element_index += blockDim.x)
    {
        ((float *)shared_output_accumulator)[element_index] = 0.0f;
    }
    __syncthreads();
    if (shared_maximum_context_length == 0u)
    {
        return;
    }
    for (candidate_base = 0u;
         candidate_base < shared_maximum_context_length;
         candidate_base += SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS)
    {
        if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS)
        {
            uint32_t candidate_index;
            uint32_t cache_slot_index;

            candidate_index = candidate_base + threadIdx.x;
            cache_slot_index = SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
            if (candidate_index < shared_maximum_context_length)
            {
                cache_slot_index =
                    SparkGlm52ResidentDecodeStageResolvePagedPrefillCacheSlot(
                        prompt_block_table,
                        sequence_index,
                        candidate_index,
                        first_block_token_offset,
                        block_token_count,
                        max_blocks_per_sequence,
                        kv_block_count,
                        cache_token_capacity);
            }
            shared_tile_cache_slots[threadIdx.x] = cache_slot_index;
        }
        __syncthreads();
        for (element_index = threadIdx.x;
             element_index <
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
             element_index += blockDim.x)
        {
            uint32_t key_dimension;
            uint32_t cache_slot_index;
            uint16_t key_value;

            tile_index =
                element_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
            key_dimension =
                element_index - (tile_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION);
            cache_slot_index = shared_tile_cache_slots[tile_index];
            key_value = 0u;
            if (cache_slot_index !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                if (kKvCacheIsFp8 != 0u)
                {
                    key_value = SparkGlm52ResidentDecodeStageFloatToBf16(
                        key_dimension <
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION
                        ? SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                            key_nope_cache_fp8_e4m3,
                            key_nope_cache_scale_f32,
                            cache_slot_index,
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION,
                            (head_index *
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                                key_dimension,
                            fp8_scale_block_size)
                        : SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                            mla_cache_fp8_e4m3,
                            mla_cache_scale_f32,
                            cache_slot_index,
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                                (key_dimension -
                                 SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION),
                            fp8_scale_block_size));
                }
                else
                {
                    key_value = key_dimension <
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION
                        ? key_nope_cache_bf16[
                            ((((uint64_t)cache_slot_index *
                               (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                              (uint64_t)head_index) *
                             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION) +
                            (uint64_t)key_dimension]
                        : mla_cache_bf16[
                            ((uint64_t)cache_slot_index *
                             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                            (uint64_t)(key_dimension -
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)];
                }
            }
            ((uint16_t *)shared_key_tile)[element_index] = key_value;
        }
        for (element_index = threadIdx.x;
             element_index <
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
             element_index += blockDim.x)
        {
            uint32_t value_dimension;
            uint32_t cache_slot_index;
            uint16_t cache_value;

            tile_index =
                element_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
            value_dimension =
                element_index - (tile_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION);
            cache_slot_index = shared_tile_cache_slots[tile_index];
            cache_value = 0u;
            if (cache_slot_index !=
                SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
            {
                if (kKvCacheIsFp8 != 0u)
                {
                    cache_value = SparkGlm52ResidentDecodeStageFloatToBf16(
                        SparkGlm52ResidentDecodeStageFp8ScaledRowLoad(
                            value_cache_fp8_e4m3,
                            value_cache_scale_f32,
                            cache_slot_index,
                            SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT *
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION,
                            (head_index *
                                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                                value_dimension,
                            fp8_scale_block_size));
                }
                else
                {
                    cache_value = value_cache_bf16[
                        ((((uint64_t)cache_slot_index *
                           (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT) +
                          (uint64_t)head_index) *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION) +
                        (uint64_t)value_dimension];
                }
            }
            ((uint16_t *)shared_value_tile)[element_index] = cache_value;
        }
        __syncthreads();
        if (warp_index == 0u)
        {
            nvcuda::wmma::fragment<
                nvcuda::wmma::accumulator, 16, 16, 16, float> score_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_a, 16, 16, 16,
                __nv_bfloat16, nvcuda::wmma::row_major> query_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 16, 16, 16,
                __nv_bfloat16, nvcuda::wmma::col_major> key_fragment;
            uint32_t slab_index;

            nvcuda::wmma::fill_fragment(score_fragment, 0.0f);
            for (slab_index = 0u;
                 slab_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION;
                 slab_index += 16u)
            {
                nvcuda::wmma::load_matrix_sync(
                    query_fragment,
                    &shared_query_tile[0][slab_index],
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION);
                nvcuda::wmma::load_matrix_sync(
                    key_fragment,
                    &shared_key_tile[0][slab_index],
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION);
                nvcuda::wmma::mma_sync(
                    score_fragment, query_fragment, key_fragment, score_fragment);
            }
            nvcuda::wmma::store_matrix_sync(
                &shared_score_tile[0][0],
                score_fragment,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS,
                nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS)
        {
            uint32_t key_index;
            uint32_t candidate_index;
            float tile_maximum;
            float next_maximum;
            float old_scale;
            float probability;
            float probability_sum;
            float masked_score;

            tile_index = threadIdx.x;
            tile_maximum = -FLT_MAX;
            for (key_index = 0u;
                 key_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS;
                 ++key_index)
            {
                candidate_index = candidate_base + key_index;
                masked_score = -FLT_MAX;
                if (shared_query_valid[tile_index] != 0u &&
                    candidate_index < shared_query_context_lengths[tile_index] &&
                    shared_tile_cache_slots[key_index] !=
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT)
                {
                    masked_score =
                        shared_score_tile[tile_index][key_index] * qk_scale;
                }
                shared_score_tile[tile_index][key_index] = masked_score;
                if (masked_score > tile_maximum)
                    tile_maximum = masked_score;
            }
            next_maximum = shared_row_maximum[tile_index] > tile_maximum
                ? shared_row_maximum[tile_index]
                : tile_maximum;
            old_scale = shared_row_denominator[tile_index] > 0.0f &&
                    next_maximum > (-FLT_MAX * 0.5f)
                ? __expf(shared_row_maximum[tile_index] - next_maximum)
                : 0.0f;
            probability_sum = 0.0f;
            for (key_index = 0u;
                 key_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS;
                 ++key_index)
            {
                probability = shared_score_tile[tile_index][key_index] >
                        (-FLT_MAX * 0.5f) &&
                        next_maximum > (-FLT_MAX * 0.5f)
                    ? __expf(shared_score_tile[tile_index][key_index] - next_maximum)
                    : 0.0f;
                shared_probability_tile[tile_index][key_index] =
                    __float2bfloat16(probability);
                probability_sum += probability;
            }
            shared_row_scale[tile_index] = old_scale;
            shared_row_denominator[tile_index] =
                (shared_row_denominator[tile_index] * old_scale) + probability_sum;
            shared_row_maximum[tile_index] = next_maximum;
        }
        __syncthreads();
        for (element_index = threadIdx.x;
             element_index <
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
             element_index += blockDim.x)
        {
            tile_index =
                element_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
            ((float *)shared_output_accumulator)[element_index] *=
                shared_row_scale[tile_index];
        }
        __syncthreads();
        {
            nvcuda::wmma::fragment<
                nvcuda::wmma::accumulator, 16, 16, 16, float> output_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_a, 16, 16, 16,
                __nv_bfloat16, nvcuda::wmma::row_major> probability_fragment;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 16, 16, 16,
                __nv_bfloat16, nvcuda::wmma::row_major> value_fragment;
            uint32_t fragment_index;
            uint32_t value_column;

            nvcuda::wmma::load_matrix_sync(
                probability_fragment,
                &shared_probability_tile[0][0],
                SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS);
            for (fragment_index = 0u;
                 fragment_index <
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_WARP_VALUE_DIMENSIONS / 16u;
                 ++fragment_index)
            {
                value_column = (warp_index *
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_WARP_VALUE_DIMENSIONS) +
                    (fragment_index * 16u);
                nvcuda::wmma::load_matrix_sync(
                    output_fragment,
                    &shared_output_accumulator[0][value_column],
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION,
                    nvcuda::wmma::mem_row_major);
                nvcuda::wmma::load_matrix_sync(
                    value_fragment,
                    &shared_value_tile[0][value_column],
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION);
                nvcuda::wmma::mma_sync(
                    output_fragment,
                    probability_fragment,
                    value_fragment,
                    output_fragment);
                nvcuda::wmma::store_matrix_sync(
                    &shared_output_accumulator[0][value_column],
                    output_fragment,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION,
                    nvcuda::wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
    for (element_index = threadIdx.x;
         element_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
         element_index += blockDim.x)
    {
        uint32_t value_dimension;
        uint64_t output_row_offset;
        float output_value;

        tile_index =
            element_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
        value_dimension =
            element_index - (tile_index *
                SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION);
        if (shared_query_valid[tile_index] == 0u)
        {
            continue;
        }
        output_row_offset =
            ((((uint64_t)sequence_index *
               (uint64_t)effective_prompt_token_stride) +
              (uint64_t)(query_tile_base + tile_index)) *
             (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT +
             (uint64_t)head_index) *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION;
        output_value = shared_row_denominator[tile_index] > 0.0f
            ? shared_output_accumulator[tile_index][value_dimension] /
                shared_row_denominator[tile_index]
            : 0.0f;
        prompt_attention_output_latent_bf16[
            output_row_offset + (uint64_t)value_dimension] =
            SparkGlm52ResidentDecodeStageFloatToBf16(output_value);
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
    uint32_t prompt_token_offset,
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
        {
            dim3 wmma_attention_grid;

            wmma_attention_grid.x = active_sequence_count;
            wmma_attention_grid.y = (prompt_token_count +
                SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS - 1u) /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_TILE_TOKENS;
            wmma_attention_grid.z = SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT;
            if (SparkGlm52ResidentDecodeStageExecutionRequiresFp8KvCacheCuda(
                    node_context))
            {
                const SparkGlm52ResidentDecodeStageFp8KvCachePlan *fp8_kv_cache_plan;

                if (!SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(
                        node_context))
                {
                    return SPARK_STATUS_INVALID_ARGUMENT;
                }
                fp8_kv_cache_plan = node_context->fp8_kv_cache_plan;
                SparkGlm52ResidentDecodeStagePagedPrefillAttentionWmmaKernel<1u><<<
                    wmma_attention_grid,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_THREADS,
                    0u,
                    typed_cuda_stream>>>(
                    (const uint16_t *)effective_prompt_query_latent_bf16,
                    (const uint16_t *)effective_prompt_rotated_query_rope_bf16,
                    0,
                    0,
                    0,
                    fp8_kv_cache_plan->mla_cache_fp8_e4m3,
                    fp8_kv_cache_plan->mla_cache_scale_f32,
                    fp8_kv_cache_plan->key_nope_cache_fp8_e4m3,
                    fp8_kv_cache_plan->key_nope_cache_scale_f32,
                    fp8_kv_cache_plan->value_cache_fp8_e4m3,
                    fp8_kv_cache_plan->value_cache_scale_f32,
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
                    node_context->qk_scale,
                    fp8_kv_cache_plan->scale_block_size);
            }
            else
            {
                SparkGlm52ResidentDecodeStagePagedPrefillAttentionWmmaKernel<0u><<<
                    wmma_attention_grid,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_WMMA_PREFILL_THREADS,
                    0u,
                    typed_cuda_stream>>>(
                    (const uint16_t *)effective_prompt_query_latent_bf16,
                    (const uint16_t *)effective_prompt_rotated_query_rope_bf16,
                    (const uint16_t *)node_context->mla_cache_bf16,
                    (const uint16_t *)node_context->key_nope_cache_bf16,
                    (const uint16_t *)node_context->value_cache_bf16,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    node_context->qk_scale,
                    0u);
            }
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
