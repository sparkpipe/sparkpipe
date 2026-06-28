#include "spark_glm52_resident_decode_stage_backend.h"

#include <cuda_runtime.h>

#ifndef SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT 1
#endif

#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
#include <cublasLt.h>
#endif

#include <float.h>
#include <stdint.h>

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_WARP_LANES 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT UINT32_MAX

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

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodePositiveE4m3Scale(
    float value)
{
    float best_delta;
    uint8_t best_value;
    uint32_t exponent;
    uint32_t mantissa;

    if (!isfinite(value) || value <= 0.0f)
    {
        return 0u;
    }
    best_value = 1u;
    best_delta = fabsf(value - ldexpf(1.0f / 8.0f, -6));
    for (exponent = 0u; exponent < 16u; ++exponent)
    {
        for (mantissa = exponent == 0u ? 1u : 0u; mantissa < 8u; ++mantissa)
        {
            float candidate;
            float delta;

            candidate = exponent == 0u
                ? ldexpf((float)mantissa / 8.0f, -6)
                : ldexpf(1.0f + ((float)mantissa / 8.0f),
                    (int32_t)exponent - 7);
            delta = fabsf(value - candidate);
            if (delta < best_delta)
            {
                best_delta = delta;
                best_value = (uint8_t)((exponent << 3u) | mantissa);
            }
        }
    }
    return best_value;
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

static __device__ __forceinline__ uint8_t SparkGlm52ResidentDecodeStageEncodeE2m1(
    float value)
{
    const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float absolute_value;
    float best_delta;
    uint8_t best_index;
    uint8_t value_index;
    uint8_t sign_bit;

    if (!isfinite(value))
    {
        return 0u;
    }
    sign_bit = value < 0.0f ? 8u : 0u;
    absolute_value = fabsf(value);
    best_index = 0u;
    best_delta = fabsf(absolute_value - values[0]);
    for (value_index = 1u; value_index < 8u; ++value_index)
    {
        float delta;

        delta = fabsf(absolute_value - values[value_index]);
        if (delta < best_delta)
        {
            best_delta = delta;
            best_index = value_index;
        }
    }
    return (uint8_t)(sign_bit | best_index);
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

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageDecodeMxfp4Weight(
    const uint8_t *payload_u8,
    const uint8_t *scale_e8m0_u8,
    uint32_t row_index,
    uint32_t hidden_index)
{
    uint64_t packed_row_stride;
    uint64_t scale_row_stride;
    uint64_t packed_index;
    uint64_t scale_index;
    uint8_t packed_value;
    uint8_t nibble;
    float scale_value;

    packed_row_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u;
    scale_row_stride =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    packed_index =
        ((uint64_t)row_index * packed_row_stride) +
        ((uint64_t)hidden_index >> 1u);
    scale_index =
        ((uint64_t)row_index * scale_row_stride) +
        ((uint64_t)hidden_index /
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE);
    packed_value = payload_u8[packed_index];
    nibble = (hidden_index & 1u) == 0u
        ? (uint8_t)(packed_value & 0x0fu)
        : (uint8_t)(packed_value >> 4u);
    scale_value = SparkGlm52ResidentDecodeStageDecodeE8m0(
        scale_e8m0_u8[scale_index]);
    return SparkGlm52ResidentDecodeStageDecodeE2m1(nibble) * scale_value;
}

static __device__ __forceinline__ float SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
    const uint8_t *payload_u8,
    const uint8_t *scale_e4m3_u8,
    uint32_t packed_row_stride,
    uint32_t scale_row_stride,
    uint32_t row_index,
    uint32_t column_index)
{
    uint64_t packed_index;
    uint64_t scale_index;
    uint8_t packed_value;
    uint8_t nibble;
    float scale_value;

    packed_index =
        ((uint64_t)row_index * (uint64_t)packed_row_stride) +
        ((uint64_t)column_index >> 1u);
    scale_index =
        ((uint64_t)row_index * (uint64_t)scale_row_stride) +
        ((uint64_t)column_index /
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE);
    packed_value = payload_u8[packed_index];
    nibble = (column_index & 1u) == 0u
        ? (uint8_t)(packed_value & 0x0fu)
        : (uint8_t)(packed_value >> 4u);
    scale_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
        scale_e4m3_u8[scale_index]);
    return SparkGlm52ResidentDecodeStageDecodeE2m1(nibble) * scale_value;
}

static __global__ void SparkGlm52ResidentDecodeStageMarkPhaseKernel(
    uint64_t *phase_clock_cycles,
    uint32_t phase_index)
{
    if (phase_clock_cycles != 0 && blockIdx.x == 0u && threadIdx.x == 0u &&
        phase_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT)
    {
        phase_clock_cycles[phase_index] = clock64();
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
        addressed_token_index /
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
    if (logical_block_index >= (uint64_t)max_blocks_per_sequence)
    {
        return SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_CACHE_SLOT;
    }
    block_token_offset =
        addressed_token_index %
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
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
        ((uint64_t)physical_block_index *
         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS) +
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
    const uint16_t *__restrict__ input_hidden_bf16,
    const uint16_t *__restrict__ projected_hidden_bf16,
    uint16_t *__restrict__ post_attention_hidden_bf16,
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

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeRouterTopKKernel(
    const uint16_t *__restrict__ input_hidden_bf16,
    const uint16_t *__restrict__ router_weight_bf16,
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
    __shared__ uint32_t shared_selected_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    __shared__ float shared_selected_weights[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint32_t sequence_index;
    uint32_t expert_index;
    uint32_t hidden_index;
    float router_logit;
    float router_score;

    sequence_index = blockIdx.x;
    expert_index = threadIdx.x;
    if (sequence_index >= active_sequence_count ||
        expert_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
    {
        return;
    }
    router_logit = 0.0f;
    if (expert_index < expert_count)
    {
        for (hidden_index = 0u;
             hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
             ++hidden_index)
        {
            router_logit +=
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    input_hidden_bf16[
                        ((uint64_t)sequence_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                        (uint64_t)hidden_index]) *
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    router_weight_bf16[
                        ((uint64_t)expert_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                        (uint64_t)hidden_index]);
        }
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
    if (threadIdx.x == 0u)
    {
        float selected_weight_sum;
        uint32_t selected_index;

        selected_weight_sum = 0.0f;
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float best_score;
            uint32_t best_expert;
            uint32_t candidate_expert;

            best_score = -FLT_MAX;
            best_expert = 0u;
            for (candidate_expert = 0u;
                 candidate_expert < expert_count;
                 ++candidate_expert)
            {
                float candidate_score;

                candidate_score = shared_choice_scores[candidate_expert];
                if (candidate_score > best_score ||
                    (candidate_score == best_score &&
                     candidate_expert < best_expert))
                {
                    best_score = candidate_score;
                    best_expert = candidate_expert;
                }
            }
            shared_selected_ids[selected_index] = best_expert;
            shared_selected_weights[selected_index] =
                shared_route_weights[best_expert];
            selected_weight_sum += shared_selected_weights[selected_index];
            shared_choice_scores[best_expert] = -FLT_MAX;
        }
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float selected_weight;
            uint64_t route_index;

            selected_weight = shared_selected_weights[selected_index];
            if (norm_topk_prob != 0u && selected_weight_sum > 0.0f)
            {
                selected_weight /= selected_weight_sum;
            }
            selected_weight *= routed_scaling_factor;
            route_index =
                ((uint64_t)sequence_index * (uint64_t)top_k) +
                (uint64_t)selected_index;
            topk_expert_ids[route_index] = shared_selected_ids[selected_index];
            topk_weights[route_index] = selected_weight;
        }
    }
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
    __shared__ uint32_t shared_selected_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    __shared__ float shared_selected_weights[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K];
    uint32_t sequence_index;
    uint32_t expert_index;
    float router_logit;
    float router_score;

    sequence_index = blockIdx.x;
    expert_index = threadIdx.x;
    if (sequence_index >= active_sequence_count ||
        expert_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
    {
        return;
    }
    if (expert_index < expert_count)
    {
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
    if (threadIdx.x == 0u)
    {
        float selected_weight_sum;
        uint32_t selected_index;

        selected_weight_sum = 0.0f;
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float best_score;
            uint32_t best_expert;
            uint32_t candidate_expert;

            best_score = -FLT_MAX;
            best_expert = 0u;
            for (candidate_expert = 0u;
                 candidate_expert < expert_count;
                 ++candidate_expert)
            {
                float candidate_score;

                candidate_score = shared_choice_scores[candidate_expert];
                if (candidate_score > best_score ||
                    (candidate_score == best_score &&
                     candidate_expert < best_expert))
                {
                    best_score = candidate_score;
                    best_expert = candidate_expert;
                }
            }
            shared_selected_ids[selected_index] = best_expert;
            shared_selected_weights[selected_index] =
                shared_route_weights[best_expert];
            selected_weight_sum += shared_selected_weights[selected_index];
            shared_choice_scores[best_expert] = -FLT_MAX;
        }
        for (selected_index = 0u; selected_index < top_k; ++selected_index)
        {
            float selected_weight;
            uint64_t route_index;

            selected_weight = shared_selected_weights[selected_index];
            if (norm_topk_prob != 0u && selected_weight_sum > 0.0f)
            {
                selected_weight /= selected_weight_sum;
            }
            selected_weight *= routed_scaling_factor;
            route_index =
                ((uint64_t)sequence_index * (uint64_t)top_k) +
                (uint64_t)selected_index;
            topk_expert_ids[route_index] = shared_selected_ids[selected_index];
            topk_weights[route_index] = selected_weight;
        }
    }
}

static __global__ __launch_bounds__(32, 4)
void SparkGlm52ResidentDecodeStageBf16ToNvfp4Kernel(
    const uint16_t *__restrict__ input_bf16,
    uint8_t *__restrict__ output_payload_u8,
    uint8_t *__restrict__ output_scale_e4m3,
    uint32_t element_count,
    float global_scale)
{
    __shared__ float shared_scale_value;
    __shared__ uint8_t shared_scale_byte;
    uint32_t block_offset;
    float local_maximum;
    float maximum_value;

    block_offset =
        blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    if (block_offset >= element_count)
    {
        return;
    }
    local_maximum = 0.0f;
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE &&
        block_offset + threadIdx.x < element_count)
    {
        local_maximum = fabsf(
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[block_offset + threadIdx.x]));
    }
    maximum_value = SparkGlm52ResidentDecodeStageWarpReduceMax(local_maximum);
    if (threadIdx.x == 0u)
    {
        float scale_value;

        scale_value = maximum_value == 0.0f
            ? 0.0f
            : ((maximum_value * global_scale) / 6.0f);
        shared_scale_byte =
            SparkGlm52ResidentDecodeStageEncodePositiveE4m3Scale(
                scale_value);
        shared_scale_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            shared_scale_byte);
        output_scale_e4m3[blockIdx.x] = shared_scale_byte;
    }
    __syncthreads();
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE / 2u)
    {
        uint32_t low_index;
        uint32_t high_index;
        float low_value;
        float high_value;
        uint8_t low_nibble;
        uint8_t high_nibble;

        low_index = block_offset + (threadIdx.x * 2u);
        high_index = low_index + 1u;
        low_value = 0.0f;
        high_value = 0.0f;
        if (low_index < element_count)
        {
            low_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[low_index]);
        }
        if (high_index < element_count)
        {
            high_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[high_index]);
        }
        low_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (low_value * global_scale) / shared_scale_value);
        high_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (high_value * global_scale) / shared_scale_value);
        output_payload_u8[low_index >> 1u] =
            (uint8_t)(low_nibble | (high_nibble << 4u));
    }
}

static __device__ __forceinline__ int32_t SparkGlm52ResidentDecodeStageFindBoundExpertSlot(
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const uint32_t *__restrict__ bound_expert_ids,
    uint32_t bound_expert_count,
    uint32_t expert_id)
{
    uint32_t expert_index;

    if (expert_id_to_bound_slot != 0 &&
        expert_id < SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT)
    {
        return expert_id_to_bound_slot[expert_id];
    }
    for (expert_index = 0u; expert_index < bound_expert_count; ++expert_index)
    {
        if (bound_expert_ids[expert_index] == expert_id)
        {
            return (int32_t)expert_index;
        }
    }
    return -1;
}

static __device__ __forceinline__ int32_t SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
    const int32_t *__restrict__ route_bound_expert_slots,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    uint32_t bound_expert_count,
    uint32_t route_index)
{
    if (route_bound_expert_slots != 0)
    {
        return route_bound_expert_slots[route_index];
    }
    return SparkGlm52ResidentDecodeStageFindBoundExpertSlot(
        expert_id_to_bound_slot,
        bound_expert_ids,
        bound_expert_count,
        topk_expert_ids[route_index]);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageResolveBoundExpertSlotsKernel(
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    int32_t *__restrict__ route_bound_expert_slots,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    uint32_t route_index;

    route_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (route_index < route_count)
    {
        route_bound_expert_slots[route_index] =
            SparkGlm52ResidentDecodeStageFindBoundExpertSlot(
                expert_id_to_bound_slot,
                bound_expert_ids,
                bound_expert_count,
                topk_expert_ids[route_index]);
        route_index += gridDim.x * blockDim.x;
    }
}

static __global__ __launch_bounds__(32, 4)
void SparkGlm52ResidentDecodeStageBf16ToNvfp4RoutesKernel(
    const uint16_t *__restrict__ input_bf16,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    const float *__restrict__ input_scale_f32,
    uint8_t *__restrict__ output_payload_u8,
    uint8_t *__restrict__ output_scale_e4m3,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    __shared__ float shared_scale_value;
    __shared__ uint8_t shared_scale_byte;
    uint32_t route_index;
    uint32_t block_offset;
    uint32_t token_index;
    int32_t bound_expert_slot;
    float local_maximum;
    float maximum_value;
    float global_scale;

    route_index = blockIdx.y;
    block_offset =
        blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    if (route_index >= route_count ||
        block_offset >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
    {
        return;
    }
    bound_expert_slot =
        SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
    global_scale = bound_expert_slot < 0 ||
        input_scale_f32[bound_expert_slot] <= 0.0f
            ? 0.0f
            : 1.0f / input_scale_f32[bound_expert_slot];
    token_index = route_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    local_maximum = 0.0f;
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE &&
        block_offset + threadIdx.x <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
        global_scale > 0.0f)
    {
        local_maximum = fabsf(
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[
                    ((uint64_t)token_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)block_offset +
                    (uint64_t)threadIdx.x]));
    }
    maximum_value = SparkGlm52ResidentDecodeStageWarpReduceMax(local_maximum);
    if (threadIdx.x == 0u)
    {
        float scale_value;

        scale_value = maximum_value == 0.0f
            ? 0.0f
            : ((maximum_value * global_scale) / 6.0f);
        shared_scale_byte =
            SparkGlm52ResidentDecodeStageEncodePositiveE4m3Scale(
                scale_value);
        shared_scale_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            shared_scale_byte);
        output_scale_e4m3[
            ((uint64_t)route_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
              SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE)) +
            (uint64_t)blockIdx.x] = shared_scale_byte;
    }
    __syncthreads();
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE / 2u)
    {
        uint32_t low_index;
        uint32_t high_index;
        float low_value;
        float high_value;
        uint8_t low_nibble;
        uint8_t high_nibble;

        low_index = block_offset + (threadIdx.x * 2u);
        high_index = low_index + 1u;
        low_value = 0.0f;
        high_value = 0.0f;
        if (low_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
            shared_scale_value > 0.0f)
        {
            low_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[
                    ((uint64_t)token_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)low_index]);
        }
        if (high_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION &&
            shared_scale_value > 0.0f)
        {
            high_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
                input_bf16[
                    ((uint64_t)token_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)high_index]);
        }
        low_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (low_value * global_scale) / shared_scale_value);
        high_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (high_value * global_scale) / shared_scale_value);
        output_payload_u8[
            ((uint64_t)route_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u)) +
            ((uint64_t)low_index >> 1u)] =
            (uint8_t)(low_nibble | (high_nibble << 4u));
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeNvfp4GateUpTop1Kernel(
    const uint8_t *__restrict__ input_payload_u8,
    const uint8_t *__restrict__ input_scale_e4m3,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint8_t *__restrict__ gate_weight_u8,
    const uint8_t *__restrict__ gate_weight_scale_e4m3,
    const uint8_t *__restrict__ up_weight_u8,
    const uint8_t *__restrict__ up_weight_scale_e4m3,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t selected_expert_id,
    float gate_alpha,
    float up_alpha)
{
    __shared__ float shared_gate[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ float shared_up[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t intermediate_index;
    uint32_t hidden_index;
    float local_gate;
    float local_up;

    intermediate_index = blockIdx.x;
    if (intermediate_index >=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION)
    {
        return;
    }
    if (topk_expert_ids[0] != selected_expert_id)
    {
        if (threadIdx.x == 0u)
        {
            gate_bf16[intermediate_index] = 0u;
            up_bf16[intermediate_index] = 0u;
        }
        return;
    }
    local_gate = 0.0f;
    local_up = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            input_payload_u8,
            input_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            0u,
            hidden_index);
        local_gate += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                gate_weight_u8,
                gate_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                intermediate_index,
                hidden_index);
        local_up += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                up_weight_u8,
                up_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                intermediate_index,
                hidden_index);
    }
    shared_gate[threadIdx.x] = local_gate;
    shared_up[threadIdx.x] = local_up;
    __syncthreads();
    for (hidden_index = blockDim.x >> 1u;
         hidden_index != 0u;
         hidden_index >>= 1u)
    {
        if (threadIdx.x < hidden_index)
        {
            shared_gate[threadIdx.x] += shared_gate[threadIdx.x + hidden_index];
            shared_up[threadIdx.x] += shared_up[threadIdx.x + hidden_index];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        gate_bf16[intermediate_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                shared_gate[0] * gate_alpha);
        up_bf16[intermediate_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(shared_up[0] * up_alpha);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeNvfp4GateUpTopKKernel(
    const uint8_t *__restrict__ input_payload_u8,
    const uint8_t *__restrict__ input_scale_e4m3,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    const uint8_t *__restrict__ gate_weight_u8,
    const uint8_t *__restrict__ gate_weight_scale_e4m3,
    const uint8_t *__restrict__ up_weight_u8,
    const uint8_t *__restrict__ up_weight_scale_e4m3,
    const float *__restrict__ gate_input_scale_f32,
    const float *__restrict__ gate_weight_scale_2_f32,
    const float *__restrict__ up_input_scale_f32,
    const float *__restrict__ up_weight_scale_2_f32,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    __shared__ float shared_gate[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ float shared_up[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t intermediate_index;
    uint32_t route_index;
    uint32_t hidden_index;
    uint32_t bound_row_index;
    int32_t bound_expert_slot;
    float local_gate;
    float local_up;

    intermediate_index = blockIdx.x;
    route_index = blockIdx.y;
    if (route_index >= route_count ||
        intermediate_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION)
    {
        return;
    }
    bound_expert_slot =
        SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
    if (bound_expert_slot < 0)
    {
        if (threadIdx.x == 0u)
        {
            gate_bf16[
                ((uint64_t)route_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                intermediate_index] = 0u;
            up_bf16[
                ((uint64_t)route_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                intermediate_index] = 0u;
        }
        return;
    }
    bound_row_index =
        ((uint32_t)bound_expert_slot *
         SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
        intermediate_index;
    local_gate = 0.0f;
    local_up = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            input_payload_u8,
            input_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            route_index,
            hidden_index);
        local_gate += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                gate_weight_u8,
                gate_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                hidden_index);
        local_up += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                up_weight_u8,
                up_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                hidden_index);
    }
    shared_gate[threadIdx.x] = local_gate;
    shared_up[threadIdx.x] = local_up;
    __syncthreads();
    for (hidden_index = blockDim.x >> 1u;
         hidden_index != 0u;
         hidden_index >>= 1u)
    {
        if (threadIdx.x < hidden_index)
        {
            shared_gate[threadIdx.x] += shared_gate[threadIdx.x + hidden_index];
            shared_up[threadIdx.x] += shared_up[threadIdx.x + hidden_index];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        uint64_t output_index;

        output_index =
            ((uint64_t)route_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
            intermediate_index;
        gate_bf16[output_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                shared_gate[0] *
                gate_input_scale_f32[bound_expert_slot] *
                gate_weight_scale_2_f32[bound_expert_slot]);
        up_bf16[output_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                shared_up[0] *
                up_input_scale_f32[bound_expert_slot] *
                up_weight_scale_2_f32[bound_expert_slot]);
    }
}

static __global__ __launch_bounds__(32, 4)
void SparkGlm52ResidentDecodeStageMoeNvfp4SiluAndQuantTop1Kernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    const uint32_t *__restrict__ topk_expert_ids,
    uint16_t *__restrict__ intermediate_bf16,
    uint8_t *__restrict__ output_payload_u8,
    uint8_t *__restrict__ output_scale_e4m3,
    uint32_t selected_expert_id,
    float global_scale)
{
    __shared__ float shared_values[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE];
    __shared__ float shared_scale_value;
    __shared__ uint8_t shared_scale_byte;
    uint32_t block_offset;
    uint32_t value_index;
    float local_value;
    float local_maximum;
    float maximum_value;

    block_offset =
        blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    if (block_offset >=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION)
    {
        return;
    }
    value_index = block_offset + threadIdx.x;
    local_value = 0.0f;
    local_maximum = 0.0f;
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE &&
        value_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION &&
        topk_expert_ids[0] == selected_expert_id)
    {
        float gate_value;
        float up_value;

        gate_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            gate_bf16[value_index]);
        up_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            up_bf16[value_index]);
        local_value =
            (gate_value / (1.0f + expf(-gate_value))) * up_value;
        local_maximum = fabsf(local_value);
        intermediate_bf16[value_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(local_value);
    }
    maximum_value = SparkGlm52ResidentDecodeStageWarpReduceMax(local_maximum);
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE)
    {
        shared_values[threadIdx.x] = local_value;
    }
    if (threadIdx.x == 0u)
    {
        float scale_value;

        scale_value = maximum_value == 0.0f
            ? 0.0f
            : ((maximum_value * global_scale) / 6.0f);
        shared_scale_byte =
            SparkGlm52ResidentDecodeStageEncodePositiveE4m3Scale(
                scale_value);
        shared_scale_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            shared_scale_byte);
        output_scale_e4m3[blockIdx.x] = shared_scale_byte;
    }
    __syncthreads();
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE / 2u)
    {
        uint32_t low_index;
        float low_value;
        float high_value;
        uint8_t low_nibble;
        uint8_t high_nibble;

        low_index = block_offset + (threadIdx.x * 2u);
        low_value = shared_values[threadIdx.x * 2u];
        high_value = shared_values[(threadIdx.x * 2u) + 1u];
        low_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (low_value * global_scale) / shared_scale_value);
        high_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (high_value * global_scale) / shared_scale_value);
        if (low_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION)
        {
            output_payload_u8[low_index >> 1u] =
                (uint8_t)(low_nibble | (high_nibble << 4u));
        }
    }
}

static __global__ __launch_bounds__(32, 4)
void SparkGlm52ResidentDecodeStageMoeNvfp4SiluAndQuantTopKKernel(
    const uint16_t *__restrict__ gate_bf16,
    const uint16_t *__restrict__ up_bf16,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    const float *__restrict__ down_input_scale_f32,
    uint16_t *__restrict__ intermediate_bf16,
    uint8_t *__restrict__ output_payload_u8,
    uint8_t *__restrict__ output_scale_e4m3,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    __shared__ float shared_values[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE];
    __shared__ float shared_scale_value;
    __shared__ uint8_t shared_scale_byte;
    uint32_t route_index;
    uint32_t block_offset;
    uint32_t value_index;
    int32_t bound_expert_slot;
    float local_value;
    float local_maximum;
    float maximum_value;
    float global_scale;

    route_index = blockIdx.y;
    block_offset =
        blockIdx.x * SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE;
    if (route_index >= route_count ||
        block_offset >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION)
    {
        return;
    }
    bound_expert_slot =
        SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
    global_scale = bound_expert_slot < 0 ||
        down_input_scale_f32[bound_expert_slot] <= 0.0f
            ? 0.0f
            : 1.0f / down_input_scale_f32[bound_expert_slot];
    value_index = block_offset + threadIdx.x;
    local_value = 0.0f;
    local_maximum = 0.0f;
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE &&
        value_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION &&
        global_scale > 0.0f)
    {
        uint64_t route_value_index;
        float gate_value;
        float up_value;

        route_value_index =
            ((uint64_t)route_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
            value_index;
        gate_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            gate_bf16[route_value_index]);
        up_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            up_bf16[route_value_index]);
        local_value =
            (gate_value / (1.0f + expf(-gate_value))) * up_value;
        local_maximum = fabsf(local_value);
        intermediate_bf16[route_value_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(local_value);
    }
    maximum_value = SparkGlm52ResidentDecodeStageWarpReduceMax(local_maximum);
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE)
    {
        shared_values[threadIdx.x] = local_value;
    }
    if (threadIdx.x == 0u)
    {
        float scale_value;

        scale_value = maximum_value == 0.0f
            ? 0.0f
            : ((maximum_value * global_scale) / 6.0f);
        shared_scale_byte =
            SparkGlm52ResidentDecodeStageEncodePositiveE4m3Scale(
                scale_value);
        shared_scale_value = SparkGlm52ResidentDecodeStageFp8E4m3ToFloat(
            shared_scale_byte);
        output_scale_e4m3[
            ((uint64_t)route_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
              SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE)) +
            (uint64_t)blockIdx.x] = shared_scale_byte;
    }
    __syncthreads();
    if (threadIdx.x < SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE / 2u)
    {
        uint32_t low_index;
        float low_value;
        float high_value;
        uint8_t low_nibble;
        uint8_t high_nibble;

        low_index = block_offset + (threadIdx.x * 2u);
        low_value = shared_values[threadIdx.x * 2u];
        high_value = shared_values[(threadIdx.x * 2u) + 1u];
        low_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (low_value * global_scale) / shared_scale_value);
        high_nibble = shared_scale_value == 0.0f
            ? 0u
            : SparkGlm52ResidentDecodeStageEncodeE2m1(
                (high_value * global_scale) / shared_scale_value);
        output_payload_u8[
            ((uint64_t)route_index *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
              2u)) +
            ((uint64_t)low_index >> 1u)] =
            (uint8_t)(low_nibble | (high_nibble << 4u));
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeNvfp4DownTop1Kernel(
    const uint8_t *__restrict__ intermediate_payload_u8,
    const uint8_t *__restrict__ intermediate_scale_e4m3,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint8_t *__restrict__ down_weight_u8,
    const uint8_t *__restrict__ down_weight_scale_e4m3,
    uint16_t *__restrict__ route_output_bf16,
    uint32_t selected_expert_id,
    float down_alpha)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t hidden_index;
    uint32_t intermediate_index;
    float local_sum;
    float output_sum;

    hidden_index = blockIdx.x;
    if (hidden_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
    {
        return;
    }
    if (topk_expert_ids[0] != selected_expert_id)
    {
        if (threadIdx.x == 0u)
        {
            route_output_bf16[hidden_index] = 0u;
        }
        return;
    }
    local_sum = 0.0f;
    for (intermediate_index = threadIdx.x;
         intermediate_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
         intermediate_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            0u,
            intermediate_index);
        local_sum += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                down_weight_u8,
                down_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                hidden_index,
                intermediate_index);
    }
    output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        route_output_bf16[hidden_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(output_sum * down_alpha);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeNvfp4DownTopKKernel(
    const uint8_t *__restrict__ intermediate_payload_u8,
    const uint8_t *__restrict__ intermediate_scale_e4m3,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    const uint8_t *__restrict__ down_weight_u8,
    const uint8_t *__restrict__ down_weight_scale_e4m3,
    const float *__restrict__ down_input_scale_f32,
    const float *__restrict__ down_weight_scale_2_f32,
    uint16_t *__restrict__ route_output_bf16,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t hidden_index;
    uint32_t route_index;
    uint32_t intermediate_index;
    uint32_t bound_row_index;
    int32_t bound_expert_slot;
    float local_sum;
    float output_sum;

    hidden_index = blockIdx.x;
    route_index = blockIdx.y;
    if (route_index >= route_count ||
        hidden_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
    {
        return;
    }
    bound_expert_slot =
        SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
    if (bound_expert_slot < 0)
    {
        if (threadIdx.x == 0u)
        {
            route_output_bf16[
                ((uint64_t)route_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                hidden_index] = 0u;
        }
        return;
    }
    bound_row_index =
        ((uint32_t)bound_expert_slot *
         SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
        hidden_index;
    local_sum = 0.0f;
    for (intermediate_index = threadIdx.x;
         intermediate_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
         intermediate_index += blockDim.x)
    {
        float input_value;

        input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            route_index,
            intermediate_index);
        local_sum += input_value *
            SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                down_weight_u8,
                down_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                intermediate_index);
    }
    output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        route_output_bf16[
            ((uint64_t)route_index *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
            hidden_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                output_sum *
                down_input_scale_f32[bound_expert_slot] *
                down_weight_scale_2_f32[bound_expert_slot]);
    }
}


static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageGroupedMoeClearRouteStateKernel(
    uint32_t *__restrict__ expert_route_counts,
    uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ expert_route_write_cursors,
    uint32_t bound_expert_count)
{
    uint32_t expert_slot_index;

    expert_slot_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (expert_slot_index <= bound_expert_count)
    {
        if (expert_slot_index < bound_expert_count)
        {
            expert_route_counts[expert_slot_index] = 0u;
            expert_route_write_cursors[expert_slot_index] = 0u;
        }
        expert_route_offsets[expert_slot_index] = 0u;
        expert_slot_index += gridDim.x * blockDim.x;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageGroupedMoeCountRoutesKernel(
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    uint32_t *__restrict__ expert_route_counts,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    uint32_t route_index;

    route_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (route_index < route_count)
    {
        int32_t bound_expert_slot;

        bound_expert_slot = SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
        if (bound_expert_slot >= 0 &&
            (uint32_t)bound_expert_slot < bound_expert_count)
        {
            atomicAdd(&expert_route_counts[(uint32_t)bound_expert_slot], 1u);
        }
        route_index += gridDim.x * blockDim.x;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoePrefixRoutesKernel(
    const uint32_t *__restrict__ expert_route_counts,
    uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ expert_route_write_cursors,
    uint32_t bound_expert_count)
{
    __shared__ uint32_t shared_counts[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT];
    uint32_t expert_slot_index;
    uint32_t running_offset;

    expert_slot_index = threadIdx.x;
    if (expert_slot_index < bound_expert_count)
    {
        shared_counts[expert_slot_index] = expert_route_counts[expert_slot_index];
        expert_route_write_cursors[expert_slot_index] = 0u;
    }
    __syncthreads();

    if (threadIdx.x == 0u)
    {
        running_offset = 0u;
        for (expert_slot_index = 0u;
             expert_slot_index < bound_expert_count;
             ++expert_slot_index)
        {
            expert_route_offsets[expert_slot_index] = running_offset;
            running_offset += shared_counts[expert_slot_index];
        }
        expert_route_offsets[bound_expert_count] = running_offset;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageGroupedMoeScatterRoutesKernel(
    const uint32_t *__restrict__ topk_expert_ids,
    const uint32_t *__restrict__ bound_expert_ids,
    const int32_t *__restrict__ expert_id_to_bound_slot,
    const int32_t *__restrict__ route_bound_expert_slots,
    const uint32_t *__restrict__ expert_route_offsets,
    uint32_t *__restrict__ expert_route_write_cursors,
    uint32_t *__restrict__ route_indices_by_expert,
    uint32_t route_count,
    uint32_t bound_expert_count)
{
    uint32_t route_index;

    route_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (route_index < route_count)
    {
        int32_t bound_expert_slot;

        bound_expert_slot = SparkGlm52ResidentDecodeStageLoadBoundExpertSlot(
            route_bound_expert_slots,
            expert_id_to_bound_slot,
            topk_expert_ids,
            bound_expert_ids,
            bound_expert_count,
            route_index);
        if (bound_expert_slot >= 0 &&
            (uint32_t)bound_expert_slot < bound_expert_count)
        {
            uint32_t local_route_index;
            uint32_t grouped_route_index;

            local_route_index = atomicAdd(
                &expert_route_write_cursors[(uint32_t)bound_expert_slot],
                1u);
            grouped_route_index =
                expert_route_offsets[(uint32_t)bound_expert_slot] +
                local_route_index;
            route_indices_by_expert[grouped_route_index] = route_index;
        }
        route_index += gridDim.x * blockDim.x;
    }
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageGroupedMoeTileRouteCount(
    uint32_t grouped_route_count,
    uint32_t route_tile_index,
    uint32_t route_tile_count)
{
    uint32_t consumed_route_count;

    consumed_route_count = route_tile_index * route_tile_count;
    if (consumed_route_count >= grouped_route_count)
    {
        return 0u;
    }
    grouped_route_count -= consumed_route_count;
    return grouped_route_count < route_tile_count
        ? grouped_route_count
        : route_tile_count;
}


static __device__ __forceinline__ uint64_t SparkGlm52ResidentDecodeStagePackGroupedMoeWorkItem(
    uint32_t bound_expert_slot,
    uint32_t route_tile_index,
    uint32_t row_index)
{
    return ((uint64_t)row_index << 32u) |
        ((uint64_t)route_tile_index << 16u) |
        (uint64_t)bound_expert_slot;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageGroupedMoeWorkItemExpertSlot(
    uint64_t work_item)
{
    return (uint32_t)(work_item & 0xffffu);
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRouteTile(
    uint64_t work_item)
{
    return (uint32_t)((work_item >> 16u) & 0xffffu);
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRow(
    uint64_t work_item)
{
    return (uint32_t)(work_item >> 32u);
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoeBuildWorkItemsKernel(
    const uint32_t *__restrict__ expert_route_counts,
    uint32_t *__restrict__ work_item_count,
    uint32_t *__restrict__ work_item_cursor,
    uint32_t *__restrict__ work_item_overflow,
    uint64_t *__restrict__ work_items,
    uint32_t bound_expert_count,
    uint32_t route_tile_count,
    uint32_t row_count,
    uint32_t maximum_work_item_count)
{
    uint32_t bound_expert_slot;
    uint32_t output_work_item_count;

    if (threadIdx.x != 0u || blockIdx.x != 0u)
    {
        return;
    }
    output_work_item_count = 0u;
    *work_item_cursor = 0u;
    *work_item_overflow = 0u;
    for (bound_expert_slot = 0u;
         bound_expert_slot < bound_expert_count;
         ++bound_expert_slot)
    {
        uint32_t grouped_route_count;
        uint32_t route_tile_index;
        uint32_t route_tile_total;

        grouped_route_count = expert_route_counts[bound_expert_slot];
        route_tile_total =
            (grouped_route_count + route_tile_count - 1u) / route_tile_count;
        for (route_tile_index = 0u;
             route_tile_index < route_tile_total;
             ++route_tile_index)
        {
            uint32_t row_index;

            for (row_index = 0u; row_index < row_count; ++row_index)
            {
                if (output_work_item_count >= maximum_work_item_count)
                {
                    *work_item_overflow = 1u;
                    *work_item_count = output_work_item_count;
                    return;
                }
                work_items[output_work_item_count] =
                    SparkGlm52ResidentDecodeStagePackGroupedMoeWorkItem(
                        bound_expert_slot,
                        route_tile_index,
                        row_index);
                output_work_item_count += 1u;
            }
        }
    }
    *work_item_count = output_work_item_count;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoeNvfp4PersistentGateUpKernel(
    const uint8_t *__restrict__ input_payload_u8,
    const uint8_t *__restrict__ input_scale_e4m3,
    const uint32_t *__restrict__ expert_route_counts,
    const uint32_t *__restrict__ expert_route_offsets,
    const uint32_t *__restrict__ work_item_count,
    uint32_t *__restrict__ work_item_cursor,
    const uint64_t *__restrict__ work_items,
    const uint32_t *__restrict__ route_indices_by_expert,
    const uint8_t *__restrict__ gate_weight_u8,
    const uint8_t *__restrict__ gate_weight_scale_e4m3,
    const uint8_t *__restrict__ up_weight_u8,
    const uint8_t *__restrict__ up_weight_scale_e4m3,
    const float *__restrict__ gate_input_scale_f32,
    const float *__restrict__ gate_weight_scale_2_f32,
    const float *__restrict__ up_input_scale_f32,
    const float *__restrict__ up_weight_scale_2_f32,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t route_tile_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ uint32_t shared_route_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    __shared__ uint32_t shared_work_item_index;
    __shared__ uint64_t shared_work_item;
    float local_gate[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    float local_up[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];

    while (true)
    {
        uint64_t work_item;
        uint32_t intermediate_index;
        uint32_t bound_expert_slot;
        uint32_t route_tile_index;
        uint32_t grouped_route_count;
        uint32_t active_route_tile_count;
        uint32_t route_tile_offset;
        uint32_t hidden_index;
        uint32_t route_tile_lane;
        uint32_t bound_row_index;

        if (threadIdx.x == 0u)
        {
            shared_work_item_index = atomicAdd(work_item_cursor, 1u);
            if (shared_work_item_index < *work_item_count)
            {
                shared_work_item = work_items[shared_work_item_index];
            }
        }
        __syncthreads();
        if (shared_work_item_index >= *work_item_count)
        {
            return;
        }
        work_item = shared_work_item;
        bound_expert_slot =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemExpertSlot(work_item);
        route_tile_index =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRouteTile(work_item);
        intermediate_index =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRow(work_item);
        grouped_route_count = expert_route_counts[bound_expert_slot];
        active_route_tile_count =
            SparkGlm52ResidentDecodeStageGroupedMoeTileRouteCount(
                grouped_route_count,
                route_tile_index,
                route_tile_count);
        route_tile_offset =
            expert_route_offsets[bound_expert_slot] +
            (route_tile_index * route_tile_count);
        if (threadIdx.x < active_route_tile_count)
        {
            shared_route_indices[threadIdx.x] =
                route_indices_by_expert[route_tile_offset + threadIdx.x];
        }
        __syncthreads();
        for (route_tile_lane = 0u;
             route_tile_lane <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT;
             ++route_tile_lane)
        {
            local_gate[route_tile_lane] = 0.0f;
            local_up[route_tile_lane] = 0.0f;
        }
        bound_row_index =
            (bound_expert_slot *
             SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
            intermediate_index;
        for (hidden_index = threadIdx.x;
             hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
             hidden_index += blockDim.x)
        {
            float gate_weight_value;
            float up_weight_value;

            gate_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                gate_weight_u8,
                gate_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                hidden_index);
            up_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                up_weight_u8,
                up_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                hidden_index);
            for (route_tile_lane = 0u;
                 route_tile_lane < active_route_tile_count;
                 ++route_tile_lane)
            {
                uint32_t route_index;
                float input_value;

                route_index = shared_route_indices[route_tile_lane];
                input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                    input_payload_u8,
                    input_scale_e4m3,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                    route_index,
                    hidden_index);
                local_gate[route_tile_lane] += input_value * gate_weight_value;
                local_up[route_tile_lane] += input_value * up_weight_value;
            }
        }
        for (route_tile_lane = 0u;
             route_tile_lane < active_route_tile_count;
             ++route_tile_lane)
        {
            float gate_sum;
            float up_sum;
            uint32_t route_index;
            uint64_t output_index;

            gate_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
                local_gate[route_tile_lane],
                shared_reduction);
            up_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
                local_up[route_tile_lane],
                shared_reduction);
            if (threadIdx.x == 0u)
            {
                route_index = shared_route_indices[route_tile_lane];
                output_index =
                    ((uint64_t)route_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                    intermediate_index;
                gate_bf16[output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(
                        gate_sum *
                        gate_input_scale_f32[bound_expert_slot] *
                        gate_weight_scale_2_f32[bound_expert_slot]);
                up_bf16[output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(
                        up_sum *
                        up_input_scale_f32[bound_expert_slot] *
                        up_weight_scale_2_f32[bound_expert_slot]);
            }
        }
        __syncthreads();
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoeNvfp4PersistentDownKernel(
    const uint8_t *__restrict__ intermediate_payload_u8,
    const uint8_t *__restrict__ intermediate_scale_e4m3,
    const uint32_t *__restrict__ expert_route_counts,
    const uint32_t *__restrict__ expert_route_offsets,
    const uint32_t *__restrict__ work_item_count,
    uint32_t *__restrict__ work_item_cursor,
    const uint64_t *__restrict__ work_items,
    const uint32_t *__restrict__ route_indices_by_expert,
    const uint8_t *__restrict__ down_weight_u8,
    const uint8_t *__restrict__ down_weight_scale_e4m3,
    const float *__restrict__ down_input_scale_f32,
    const float *__restrict__ down_weight_scale_2_f32,
    uint16_t *__restrict__ route_output_bf16,
    uint32_t route_tile_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ uint32_t shared_route_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    __shared__ uint32_t shared_work_item_index;
    __shared__ uint64_t shared_work_item;
    float local_sum[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];

    while (true)
    {
        uint64_t work_item;
        uint32_t hidden_index;
        uint32_t bound_expert_slot;
        uint32_t route_tile_index;
        uint32_t grouped_route_count;
        uint32_t active_route_tile_count;
        uint32_t route_tile_offset;
        uint32_t intermediate_index;
        uint32_t route_tile_lane;
        uint32_t bound_row_index;

        if (threadIdx.x == 0u)
        {
            shared_work_item_index = atomicAdd(work_item_cursor, 1u);
            if (shared_work_item_index < *work_item_count)
            {
                shared_work_item = work_items[shared_work_item_index];
            }
        }
        __syncthreads();
        if (shared_work_item_index >= *work_item_count)
        {
            return;
        }
        work_item = shared_work_item;
        bound_expert_slot =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemExpertSlot(work_item);
        route_tile_index =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRouteTile(work_item);
        hidden_index =
            SparkGlm52ResidentDecodeStageGroupedMoeWorkItemRow(work_item);
        grouped_route_count = expert_route_counts[bound_expert_slot];
        active_route_tile_count =
            SparkGlm52ResidentDecodeStageGroupedMoeTileRouteCount(
                grouped_route_count,
                route_tile_index,
                route_tile_count);
        route_tile_offset =
            expert_route_offsets[bound_expert_slot] +
            (route_tile_index * route_tile_count);
        if (threadIdx.x < active_route_tile_count)
        {
            shared_route_indices[threadIdx.x] =
                route_indices_by_expert[route_tile_offset + threadIdx.x];
        }
        __syncthreads();
        for (route_tile_lane = 0u;
             route_tile_lane <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT;
             ++route_tile_lane)
        {
            local_sum[route_tile_lane] = 0.0f;
        }
        bound_row_index =
            (bound_expert_slot * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
            hidden_index;
        for (intermediate_index = threadIdx.x;
             intermediate_index <
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
             intermediate_index += blockDim.x)
        {
            float down_weight_value;

            down_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                down_weight_u8,
                down_weight_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                bound_row_index,
                intermediate_index);
            for (route_tile_lane = 0u;
                 route_tile_lane < active_route_tile_count;
                 ++route_tile_lane)
            {
                uint32_t route_index;
                float input_value;

                route_index = shared_route_indices[route_tile_lane];
                input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                    intermediate_payload_u8,
                    intermediate_scale_e4m3,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                    route_index,
                    intermediate_index);
                local_sum[route_tile_lane] += input_value * down_weight_value;
            }
        }
        for (route_tile_lane = 0u;
             route_tile_lane < active_route_tile_count;
             ++route_tile_lane)
        {
            float output_sum;
            uint32_t route_index;
            uint64_t output_index;

            output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
                local_sum[route_tile_lane],
                shared_reduction);
            if (threadIdx.x == 0u)
            {
                route_index = shared_route_indices[route_tile_lane];
                output_index =
                    ((uint64_t)route_index *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    hidden_index;
                route_output_bf16[output_index] =
                    SparkGlm52ResidentDecodeStageFloatToBf16(
                        output_sum *
                        down_input_scale_f32[bound_expert_slot] *
                        down_weight_scale_2_f32[bound_expert_slot]);
            }
        }
        __syncthreads();
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoeNvfp4GateUpKernel(
    const uint8_t *__restrict__ input_payload_u8,
    const uint8_t *__restrict__ input_scale_e4m3,
    const uint32_t *__restrict__ expert_route_counts,
    const uint32_t *__restrict__ expert_route_offsets,
    const uint32_t *__restrict__ route_indices_by_expert,
    const uint8_t *__restrict__ gate_weight_u8,
    const uint8_t *__restrict__ gate_weight_scale_e4m3,
    const uint8_t *__restrict__ up_weight_u8,
    const uint8_t *__restrict__ up_weight_scale_e4m3,
    const float *__restrict__ gate_input_scale_f32,
    const float *__restrict__ gate_weight_scale_2_f32,
    const float *__restrict__ up_input_scale_f32,
    const float *__restrict__ up_weight_scale_2_f32,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t bound_expert_count,
    uint32_t route_tile_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ uint32_t shared_route_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    float local_gate[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    float local_up[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    uint32_t intermediate_index;
    uint32_t bound_expert_slot;
    uint32_t route_tile_index;
    uint32_t grouped_route_count;
    uint32_t active_route_tile_count;
    uint32_t route_tile_offset;
    uint32_t hidden_index;
    uint32_t route_tile_lane;
    uint32_t bound_row_index;

    intermediate_index = blockIdx.x;
    bound_expert_slot = blockIdx.y;
    route_tile_index = blockIdx.z;
    if (intermediate_index >=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        bound_expert_slot >= bound_expert_count ||
        route_tile_count == 0u ||
        route_tile_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT)
    {
        return;
    }

    grouped_route_count = expert_route_counts[bound_expert_slot];
    active_route_tile_count =
        SparkGlm52ResidentDecodeStageGroupedMoeTileRouteCount(
            grouped_route_count,
            route_tile_index,
            route_tile_count);
    if (active_route_tile_count == 0u)
    {
        return;
    }
    route_tile_offset =
        expert_route_offsets[bound_expert_slot] +
        (route_tile_index * route_tile_count);
    if (threadIdx.x < active_route_tile_count)
    {
        shared_route_indices[threadIdx.x] =
            route_indices_by_expert[route_tile_offset + threadIdx.x];
    }
    __syncthreads();

    for (route_tile_lane = 0u;
         route_tile_lane <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT;
         ++route_tile_lane)
    {
        local_gate[route_tile_lane] = 0.0f;
        local_up[route_tile_lane] = 0.0f;
    }
    bound_row_index =
        (bound_expert_slot *
         SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
        intermediate_index;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float gate_weight_value;
        float up_weight_value;

        gate_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            gate_weight_u8,
            gate_weight_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            bound_row_index,
            hidden_index);
        up_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            up_weight_u8,
            up_weight_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            bound_row_index,
            hidden_index);
        for (route_tile_lane = 0u;
             route_tile_lane < active_route_tile_count;
             ++route_tile_lane)
        {
            uint32_t route_index;
            float input_value;

            route_index = shared_route_indices[route_tile_lane];
            input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                input_payload_u8,
                input_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                route_index,
                hidden_index);
            local_gate[route_tile_lane] += input_value * gate_weight_value;
            local_up[route_tile_lane] += input_value * up_weight_value;
        }
    }

    for (route_tile_lane = 0u;
         route_tile_lane < active_route_tile_count;
         ++route_tile_lane)
    {
        float gate_sum;
        float up_sum;
        uint32_t route_index;
        uint64_t output_index;

        gate_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
            local_gate[route_tile_lane],
            shared_reduction);
        up_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
            local_up[route_tile_lane],
            shared_reduction);
        if (threadIdx.x == 0u)
        {
            route_index = shared_route_indices[route_tile_lane];
            output_index =
                ((uint64_t)route_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION) +
                intermediate_index;
            gate_bf16[output_index] =
                SparkGlm52ResidentDecodeStageFloatToBf16(
                    gate_sum *
                    gate_input_scale_f32[bound_expert_slot] *
                    gate_weight_scale_2_f32[bound_expert_slot]);
            up_bf16[output_index] =
                SparkGlm52ResidentDecodeStageFloatToBf16(
                    up_sum *
                    up_input_scale_f32[bound_expert_slot] *
                    up_weight_scale_2_f32[bound_expert_slot]);
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 1)
void SparkGlm52ResidentDecodeStageGroupedMoeNvfp4DownKernel(
    const uint8_t *__restrict__ intermediate_payload_u8,
    const uint8_t *__restrict__ intermediate_scale_e4m3,
    const uint32_t *__restrict__ expert_route_counts,
    const uint32_t *__restrict__ expert_route_offsets,
    const uint32_t *__restrict__ route_indices_by_expert,
    const uint8_t *__restrict__ down_weight_u8,
    const uint8_t *__restrict__ down_weight_scale_e4m3,
    const float *__restrict__ down_input_scale_f32,
    const float *__restrict__ down_weight_scale_2_f32,
    uint16_t *__restrict__ route_output_bf16,
    uint32_t bound_expert_count,
    uint32_t route_tile_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ uint32_t shared_route_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    float local_sum[SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT];
    uint32_t hidden_index;
    uint32_t bound_expert_slot;
    uint32_t route_tile_index;
    uint32_t grouped_route_count;
    uint32_t active_route_tile_count;
    uint32_t route_tile_offset;
    uint32_t intermediate_index;
    uint32_t route_tile_lane;
    uint32_t bound_row_index;

    hidden_index = blockIdx.x;
    bound_expert_slot = blockIdx.y;
    route_tile_index = blockIdx.z;
    if (hidden_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION ||
        bound_expert_slot >= bound_expert_count ||
        route_tile_count == 0u ||
        route_tile_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT)
    {
        return;
    }

    grouped_route_count = expert_route_counts[bound_expert_slot];
    active_route_tile_count =
        SparkGlm52ResidentDecodeStageGroupedMoeTileRouteCount(
            grouped_route_count,
            route_tile_index,
            route_tile_count);
    if (active_route_tile_count == 0u)
    {
        return;
    }
    route_tile_offset =
        expert_route_offsets[bound_expert_slot] +
        (route_tile_index * route_tile_count);
    if (threadIdx.x < active_route_tile_count)
    {
        shared_route_indices[threadIdx.x] =
            route_indices_by_expert[route_tile_offset + threadIdx.x];
    }
    __syncthreads();

    for (route_tile_lane = 0u;
         route_tile_lane <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT;
         ++route_tile_lane)
    {
        local_sum[route_tile_lane] = 0.0f;
    }
    bound_row_index =
        (bound_expert_slot * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
        hidden_index;
    for (intermediate_index = threadIdx.x;
         intermediate_index <
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
         intermediate_index += blockDim.x)
    {
        float down_weight_value;

        down_weight_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
            down_weight_u8,
            down_weight_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            bound_row_index,
            intermediate_index);
        for (route_tile_lane = 0u;
             route_tile_lane < active_route_tile_count;
             ++route_tile_lane)
        {
            uint32_t route_index;
            float input_value;

            route_index = shared_route_indices[route_tile_lane];
            input_value = SparkGlm52ResidentDecodeStageDecodeNvfp4Value(
                intermediate_payload_u8,
                intermediate_scale_e4m3,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u,
                SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
                route_index,
                intermediate_index);
            local_sum[route_tile_lane] += input_value * down_weight_value;
        }
    }

    for (route_tile_lane = 0u;
         route_tile_lane < active_route_tile_count;
         ++route_tile_lane)
    {
        float output_sum;
        uint32_t route_index;
        uint64_t output_index;

        output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
            local_sum[route_tile_lane],
            shared_reduction);
        if (threadIdx.x == 0u)
        {
            route_index = shared_route_indices[route_tile_lane];
            output_index =
                ((uint64_t)route_index *
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                hidden_index;
            route_output_bf16[output_index] =
                SparkGlm52ResidentDecodeStageFloatToBf16(
                    output_sum *
                    down_input_scale_f32[bound_expert_slot] *
                    down_weight_scale_2_f32[bound_expert_slot]);
        }
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageMoeGateUpKernel(
    const uint16_t *__restrict__ input_hidden_bf16,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint16_t *__restrict__ gate_weight_bf16,
    const uint16_t *__restrict__ up_weight_bf16,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t route_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension,
    uint32_t first_bound_expert_id,
    uint32_t bound_expert_count)
{
    __shared__ float shared_gate[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ float shared_up[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t intermediate_index;
    uint32_t route_index;
    uint32_t token_index;
    uint32_t expert_index;
    uint32_t bound_expert_index;
    uint32_t hidden_index;
    float local_gate;
    float local_up;
    float gate_sum;
    float up_sum;

    intermediate_index = blockIdx.x;
    route_index = blockIdx.y;
    if (route_index >= route_count ||
        intermediate_index >= intermediate_dimension)
    {
        return;
    }
    expert_index = topk_expert_ids[route_index];
    if (expert_index < first_bound_expert_id ||
        expert_index >= first_bound_expert_id + bound_expert_count)
    {
        if (threadIdx.x == 0u)
        {
            gate_bf16[((uint64_t)route_index * intermediate_dimension) +
                intermediate_index] = 0u;
            up_bf16[((uint64_t)route_index * intermediate_dimension) +
                intermediate_index] = 0u;
        }
        return;
    }
    token_index =
        route_index / SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    bound_expert_index = expert_index - first_bound_expert_id;
    local_gate = 0.0f;
    local_up = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < hidden_dimension;
         hidden_index += blockDim.x)
    {
        uint64_t input_offset;
        uint64_t weight_offset;
        float input_value;

        input_offset =
            ((uint64_t)token_index * (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index;
        weight_offset =
            ((((uint64_t)bound_expert_index *
               (uint64_t)intermediate_dimension) +
              (uint64_t)intermediate_index) *
             (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index;
        input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_hidden_bf16[input_offset]);
        local_gate += input_value *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                gate_weight_bf16[weight_offset]);
        local_up += input_value *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                up_weight_bf16[weight_offset]);
    }
    shared_gate[threadIdx.x] = local_gate;
    shared_up[threadIdx.x] = local_up;
    __syncthreads();
    for (hidden_index = blockDim.x >> 1u;
         hidden_index != 0u;
         hidden_index >>= 1u)
    {
        if (threadIdx.x < hidden_index)
        {
            shared_gate[threadIdx.x] += shared_gate[threadIdx.x + hidden_index];
            shared_up[threadIdx.x] += shared_up[threadIdx.x + hidden_index];
        }
        __syncthreads();
    }
    gate_sum = shared_gate[0];
    up_sum = shared_up[0];
    if (threadIdx.x == 0u)
    {
        gate_bf16[((uint64_t)route_index * intermediate_dimension) +
            intermediate_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(gate_sum);
        up_bf16[((uint64_t)route_index * intermediate_dimension) +
            intermediate_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(up_sum);
    }
}

static __global__ void SparkGlm52ResidentDecodeStageMoeSiluKernel(
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
void SparkGlm52ResidentDecodeStageMoeDownKernel(
    const uint16_t *__restrict__ intermediate_bf16,
    const uint32_t *__restrict__ topk_expert_ids,
    const uint16_t *__restrict__ down_weight_bf16,
    uint16_t *__restrict__ route_output_bf16,
    uint32_t route_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension,
    uint32_t first_bound_expert_id,
    uint32_t bound_expert_count)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t hidden_index;
    uint32_t route_index;
    uint32_t expert_index;
    uint32_t bound_expert_index;
    uint32_t intermediate_index;
    float local_sum;
    float output_sum;

    hidden_index = blockIdx.x;
    route_index = blockIdx.y;
    if (route_index >= route_count || hidden_index >= hidden_dimension)
    {
        return;
    }
    expert_index = topk_expert_ids[route_index];
    if (expert_index < first_bound_expert_id ||
        expert_index >= first_bound_expert_id + bound_expert_count)
    {
        if (threadIdx.x == 0u)
        {
            route_output_bf16[
                ((uint64_t)route_index * hidden_dimension) +
                hidden_index] = 0u;
        }
        return;
    }
    bound_expert_index = expert_index - first_bound_expert_id;
    local_sum = 0.0f;
    for (intermediate_index = threadIdx.x;
         intermediate_index < intermediate_dimension;
         intermediate_index += blockDim.x)
    {
        uint64_t input_offset;
        uint64_t weight_offset;

        input_offset =
            ((uint64_t)route_index * (uint64_t)intermediate_dimension) +
            (uint64_t)intermediate_index;
        weight_offset =
            ((((uint64_t)bound_expert_index *
               (uint64_t)hidden_dimension) +
              (uint64_t)hidden_index) *
             (uint64_t)intermediate_dimension) +
            (uint64_t)intermediate_index;
        local_sum += SparkGlm52ResidentDecodeStageBf16ToFloat(
            intermediate_bf16[input_offset]) *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                down_weight_bf16[weight_offset]);
    }
    output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        route_output_bf16[((uint64_t)route_index * hidden_dimension) +
            hidden_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(output_sum);
    }
}

static __global__ void SparkGlm52ResidentDecodeStageMoeCombineKernel(
    const uint16_t *__restrict__ residual_hidden_bf16,
    const uint16_t *__restrict__ route_output_bf16,
    const float *__restrict__ topk_weights,
    uint16_t *__restrict__ layer_output_bf16,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t top_k)
{
    uint64_t element_index;
    uint64_t element_count;
    uint64_t element_stride;

    element_count = (uint64_t)active_sequence_count *
        (uint64_t)hidden_dimension;
    element_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    element_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    while (element_index < element_count)
    {
        uint32_t token_index;
        uint32_t hidden_index;
        uint32_t topk_index;
        float output_value;

        token_index = (uint32_t)(element_index / hidden_dimension);
        hidden_index = (uint32_t)(element_index % hidden_dimension);
        output_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            residual_hidden_bf16[element_index]);
        for (topk_index = 0u; topk_index < top_k; ++topk_index)
        {
            uint64_t route_index;

            route_index =
                ((uint64_t)token_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K) +
                (uint64_t)topk_index;
            output_value += topk_weights[route_index] *
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    route_output_bf16[
                        (route_index * (uint64_t)hidden_dimension) +
                        (uint64_t)hidden_index]);
        }
        layer_output_bf16[element_index] =
            SparkGlm52ResidentDecodeStageFloatToBf16(output_value);
        element_index += element_stride;
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageDenseGateUpKernel(
    const uint16_t *__restrict__ input_hidden_bf16,
    const uint16_t *__restrict__ gate_weight_bf16,
    const uint16_t *__restrict__ up_weight_bf16,
    uint16_t *__restrict__ gate_bf16,
    uint16_t *__restrict__ up_bf16,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension)
{
    __shared__ float shared_gate[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    __shared__ float shared_up[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t intermediate_index;
    uint32_t sequence_index;
    uint32_t hidden_index;
    float local_gate;
    float local_up;

    intermediate_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        intermediate_index >= intermediate_dimension)
    {
        return;
    }
    local_gate = 0.0f;
    local_up = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < hidden_dimension;
         hidden_index += blockDim.x)
    {
        uint64_t input_offset;
        uint64_t weight_offset;
        float input_value;

        input_offset =
            ((uint64_t)sequence_index * (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index;
        weight_offset =
            ((uint64_t)intermediate_index * (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index;
        input_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            input_hidden_bf16[input_offset]);
        local_gate += input_value *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                gate_weight_bf16[weight_offset]);
        local_up += input_value *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                up_weight_bf16[weight_offset]);
    }
    shared_gate[threadIdx.x] = local_gate;
    shared_up[threadIdx.x] = local_up;
    __syncthreads();
    for (hidden_index = blockDim.x >> 1u;
         hidden_index != 0u;
         hidden_index >>= 1u)
    {
        if (threadIdx.x < hidden_index)
        {
            shared_gate[threadIdx.x] += shared_gate[threadIdx.x + hidden_index];
            shared_up[threadIdx.x] += shared_up[threadIdx.x + hidden_index];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        uint64_t output_offset;

        output_offset =
            ((uint64_t)sequence_index * (uint64_t)intermediate_dimension) +
            (uint64_t)intermediate_index;
        gate_bf16[output_offset] =
            SparkGlm52ResidentDecodeStageFloatToBf16(shared_gate[0]);
        up_bf16[output_offset] =
            SparkGlm52ResidentDecodeStageFloatToBf16(shared_up[0]);
    }
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS, 2)
void SparkGlm52ResidentDecodeStageDenseDownResidualKernel(
    const uint16_t *__restrict__ residual_hidden_bf16,
    const uint16_t *__restrict__ intermediate_bf16,
    const uint16_t *__restrict__ down_weight_bf16,
    uint16_t *__restrict__ layer_output_bf16,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t intermediate_dimension)
{
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS];
    uint32_t hidden_index;
    uint32_t sequence_index;
    uint32_t intermediate_index;
    float local_sum;
    float output_sum;
    uint64_t output_offset;

    hidden_index = blockIdx.x;
    sequence_index = blockIdx.y;
    if (sequence_index >= active_sequence_count ||
        hidden_index >= hidden_dimension)
    {
        return;
    }
    local_sum = 0.0f;
    for (intermediate_index = threadIdx.x;
         intermediate_index < intermediate_dimension;
         intermediate_index += blockDim.x)
    {
        uint64_t input_offset;
        uint64_t weight_offset;

        input_offset =
            ((uint64_t)sequence_index * (uint64_t)intermediate_dimension) +
            (uint64_t)intermediate_index;
        weight_offset =
            ((uint64_t)hidden_index * (uint64_t)intermediate_dimension) +
            (uint64_t)intermediate_index;
        local_sum += SparkGlm52ResidentDecodeStageBf16ToFloat(
            intermediate_bf16[input_offset]) *
            SparkGlm52ResidentDecodeStageBf16ToFloat(
                down_weight_bf16[weight_offset]);
    }
    output_sum = SparkGlm52ResidentDecodeStageBlockReduceSum(
        local_sum,
        shared_reduction);
    if (threadIdx.x == 0u)
    {
        output_offset =
            ((uint64_t)sequence_index * (uint64_t)hidden_dimension) +
            (uint64_t)hidden_index;
        layer_output_bf16[output_offset] =
            SparkGlm52ResidentDecodeStageFloatToBf16(
                SparkGlm52ResidentDecodeStageBf16ToFloat(
                    residual_hidden_bf16[output_offset]) +
                output_sum);
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
void SparkGlm52ResidentDecodeStageMtpDraftLogitsKernel(
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
    uint32_t hidden_index;
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
    local_sum = 0.0f;
    for (hidden_index = threadIdx.x;
         hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         hidden_index += blockDim.x)
    {
        float activation_value;
        float weight_value;

        activation_value = SparkGlm52ResidentDecodeStageBf16ToFloat(
            mtp_draft_hidden_bf16[
                ((((uint64_t)sequence_index *
                   (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
                  (uint64_t)draft_index) *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                (uint64_t)hidden_index]);
        weight_value = SparkGlm52ResidentDecodeStageDecodeMxfp4Weight(
            mtp_weight_payload_u8,
            mtp_weight_scale_e8m0_u8,
            restricted_index,
            hidden_index);
        local_sum += activation_value * weight_value;
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

typedef SparkStatus (*SparkGlm52ResidentDecodeStageGroupedMoeLaunchFunction)(
    const SparkGlm52ResidentDecodeStageGroupedMoePlan *grouped_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
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

static const SparkGlm52ResidentDecodeStageLinearPlan *SparkGlm52ResidentDecodeStageGetLinearPlan(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t plan_index,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t active_sequence_count);

static SparkStatus SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plan,
    const void *input,
    const void *weight,
    void *output,
    uint32_t active_sequence_count,
    cudaStream_t cuda_stream);

extern "C" SparkStatus SparkGlm52ResidentDecodeStageLaunchPersistentGroupedNvfp4Moe(
    const SparkGlm52ResidentDecodeStageGroupedMoePlan *grouped_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream_pointer)
{
    cudaStream_t cuda_stream;
    uint32_t route_count;
    uint32_t route_block_count;
    uint32_t bound_expert_count;
    uint32_t route_tile_count;
    uint8_t *hidden_payload_u8;
    uint8_t *hidden_scale_e4m3;
    uint8_t *intermediate_payload_u8;
    uint8_t *intermediate_scale_e4m3;

    if (grouped_moe_plan == 0 ||
        node_context == 0 ||
        pipeline_slot == 0 ||
        cuda_stream_pointer == 0 ||
        grouped_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_ABI_VERSION ||
        grouped_moe_plan->plan_kind !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_KIND_PERSISTENT_NVFP4_TOPK ||
        grouped_moe_plan->weight_format !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_WEIGHT_FORMAT_NVFP4_E2M1_E4M3 ||
        grouped_moe_plan->launch_function == 0 ||
        grouped_moe_plan->expert_route_counts == 0 ||
        grouped_moe_plan->expert_route_offsets == 0 ||
        grouped_moe_plan->expert_route_write_cursors == 0 ||
        grouped_moe_plan->route_indices_by_expert == 0 ||
        grouped_moe_plan->work_item_count == 0 ||
        grouped_moe_plan->work_item_cursor == 0 ||
        grouped_moe_plan->work_item_overflow == 0 ||
        grouped_moe_plan->work_items == 0 ||
        grouped_moe_plan->persistent_worker_block_count == 0u ||
        grouped_moe_plan->maximum_active_sequence_count < active_sequence_count ||
        grouped_moe_plan->expert_count != node_context->moe_expert_count ||
        grouped_moe_plan->top_k != node_context->moe_top_k ||
        grouped_moe_plan->intermediate_dimension !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION ||
        grouped_moe_plan->route_tile_count == 0u ||
        grouped_moe_plan->route_tile_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT ||
        node_context->layer_progression_mode !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK ||
        node_context->moe_nvfp4_bound_expert_count == 0u ||
        node_context->moe_nvfp4_bound_expert_count > node_context->moe_expert_count ||
        active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    route_count = active_sequence_count * node_context->moe_top_k;
    if (route_count > grouped_moe_plan->maximum_route_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    bound_expert_count = node_context->moe_nvfp4_bound_expert_count;
    route_tile_count = grouped_moe_plan->route_tile_count;
    if (grouped_moe_plan->maximum_work_item_count <
        route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cuda_stream = (cudaStream_t)cuda_stream_pointer;
    route_block_count = SparkGlm52ResidentDecodeStageElementBlockCount(
        route_count > bound_expert_count ? route_count : bound_expert_count + 1u);

    if (pipeline_slot->moe_router_logits != 0)
    {
        const SparkGlm52ResidentDecodeStageLinearPlan *router_plan;
        SparkStatus router_status;

        router_plan = SparkGlm52ResidentDecodeStageGetLinearPlan(
            node_context,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,
            active_sequence_count);
        if (router_plan == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        router_status = SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
            router_plan,
            pipeline_slot->post_attention_normalized_hidden_bf16,
            node_context->moe_router_weight_bf16,
            pipeline_slot->moe_router_logits,
            active_sequence_count,
            cuda_stream);
        if (router_status != SPARK_STATUS_OK)
        {
            return router_status;
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
    }
    else
    {
        SparkGlm52ResidentDecodeStageMoeRouterTopKKernel<<<
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            (const uint16_t *)node_context->moe_router_weight_bf16,
            node_context->moe_router_score_bias_f32,
            pipeline_slot->moe_topk_expert_ids,
            pipeline_slot->moe_topk_weights,
            active_sequence_count,
            node_context->moe_expert_count,
            node_context->moe_top_k,
            node_context->moe_norm_topk_prob,
            node_context->moe_routed_scaling_factor);
    }

    SparkGlm52ResidentDecodeStageResolveBoundExpertSlotsKernel<<<
        route_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        pipeline_slot->moe_topk_expert_ids,
        node_context->moe_nvfp4_bound_expert_ids,
        node_context->moe_nvfp4_expert_id_to_bound_slot,
        pipeline_slot->moe_bound_expert_slots,
        route_count,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoeClearRouteStateKernel<<<
        route_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->expert_route_offsets,
        grouped_moe_plan->expert_route_write_cursors,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoeCountRoutesKernel<<<
        route_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        pipeline_slot->moe_topk_expert_ids,
        node_context->moe_nvfp4_bound_expert_ids,
        node_context->moe_nvfp4_expert_id_to_bound_slot,
        pipeline_slot->moe_bound_expert_slots,
        grouped_moe_plan->expert_route_counts,
        route_count,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoePrefixRoutesKernel<<<
        1u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->expert_route_offsets,
        grouped_moe_plan->expert_route_write_cursors,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoeScatterRoutesKernel<<<
        route_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        pipeline_slot->moe_topk_expert_ids,
        node_context->moe_nvfp4_bound_expert_ids,
        node_context->moe_nvfp4_expert_id_to_bound_slot,
        pipeline_slot->moe_bound_expert_slots,
        grouped_moe_plan->expert_route_offsets,
        grouped_moe_plan->expert_route_write_cursors,
        grouped_moe_plan->route_indices_by_expert,
        route_count,
        bound_expert_count);

    hidden_payload_u8 = (uint8_t *)pipeline_slot->moe_route_output_bf16;
    hidden_scale_e4m3 =
        hidden_payload_u8 +
        ((uint64_t)route_count *
         (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u));
    SparkGlm52ResidentDecodeStageBf16ToNvfp4RoutesKernel<<<
        dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            route_count,
            1u),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
        pipeline_slot->moe_topk_expert_ids,
        node_context->moe_nvfp4_bound_expert_ids,
        node_context->moe_nvfp4_expert_id_to_bound_slot,
        pipeline_slot->moe_bound_expert_slots,
        node_context->moe_nvfp4_gate_input_scale_f32,
        hidden_payload_u8,
        hidden_scale_e4m3,
        route_count,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoeBuildWorkItemsKernel<<<
        1u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->work_item_count,
        grouped_moe_plan->work_item_cursor,
        grouped_moe_plan->work_item_overflow,
        grouped_moe_plan->work_items,
        bound_expert_count,
        route_tile_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
        grouped_moe_plan->maximum_work_item_count);

    SparkGlm52ResidentDecodeStageGroupedMoeNvfp4PersistentGateUpKernel<<<
        grouped_moe_plan->persistent_worker_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        hidden_payload_u8,
        hidden_scale_e4m3,
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->expert_route_offsets,
        grouped_moe_plan->work_item_count,
        grouped_moe_plan->work_item_cursor,
        grouped_moe_plan->work_items,
        grouped_moe_plan->route_indices_by_expert,
        node_context->moe_nvfp4_gate_weight_u8,
        node_context->moe_nvfp4_gate_weight_scale_e4m3,
        node_context->moe_nvfp4_up_weight_u8,
        node_context->moe_nvfp4_up_weight_scale_e4m3,
        node_context->moe_nvfp4_gate_input_scale_f32,
        node_context->moe_nvfp4_gate_weight_scale_2_f32,
        node_context->moe_nvfp4_up_input_scale_f32,
        node_context->moe_nvfp4_up_weight_scale_2_f32,
        (uint16_t *)pipeline_slot->moe_gate_bf16,
        (uint16_t *)pipeline_slot->moe_up_bf16,
        route_tile_count);

    intermediate_payload_u8 = (uint8_t *)pipeline_slot->moe_intermediate_bf16;
    intermediate_scale_e4m3 =
        intermediate_payload_u8 +
        ((uint64_t)route_count *
         (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION / 2u));
    SparkGlm52ResidentDecodeStageMoeNvfp4SiluAndQuantTopKKernel<<<
        dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            route_count,
            1u),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->moe_gate_bf16,
        (const uint16_t *)pipeline_slot->moe_up_bf16,
        pipeline_slot->moe_topk_expert_ids,
        node_context->moe_nvfp4_bound_expert_ids,
        node_context->moe_nvfp4_expert_id_to_bound_slot,
        pipeline_slot->moe_bound_expert_slots,
        node_context->moe_nvfp4_down_input_scale_f32,
        (uint16_t *)pipeline_slot->moe_intermediate_bf16,
        intermediate_payload_u8,
        intermediate_scale_e4m3,
        route_count,
        bound_expert_count);

    SparkGlm52ResidentDecodeStageGroupedMoeBuildWorkItemsKernel<<<
        1u,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->work_item_count,
        grouped_moe_plan->work_item_cursor,
        grouped_moe_plan->work_item_overflow,
        grouped_moe_plan->work_items,
        bound_expert_count,
        route_tile_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        grouped_moe_plan->maximum_work_item_count);

    SparkGlm52ResidentDecodeStageGroupedMoeNvfp4PersistentDownKernel<<<
        grouped_moe_plan->persistent_worker_block_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        intermediate_payload_u8,
        intermediate_scale_e4m3,
        grouped_moe_plan->expert_route_counts,
        grouped_moe_plan->expert_route_offsets,
        grouped_moe_plan->work_item_count,
        grouped_moe_plan->work_item_cursor,
        grouped_moe_plan->work_items,
        grouped_moe_plan->route_indices_by_expert,
        node_context->moe_nvfp4_down_weight_u8,
        node_context->moe_nvfp4_down_weight_scale_e4m3,
        node_context->moe_nvfp4_down_input_scale_f32,
        node_context->moe_nvfp4_down_weight_scale_2_f32,
        (uint16_t *)pipeline_slot->moe_route_output_bf16,
        route_tile_count);

    SparkGlm52ResidentDecodeStageMoeCombineKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(
            (uint64_t)active_sequence_count *
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)pipeline_slot->moe_route_output_bf16,
        pipeline_slot->moe_topk_weights,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->moe_top_k);

    return SPARK_STATUS_OK;
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
    if (linear_plan == 0 || input == 0 || weight == 0 || output == 0)
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
#if SPARK_GLM52_RESIDENT_DECODE_STAGE_ENABLE_CUBLASLT
    if (linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR ||
        linear_plan->plan_kind ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR)
    {
        cublasStatus_t cublas_status;
        float alpha;
        float beta;

        if (linear_plan->cublaslt_handle == 0 ||
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
        return plan_is_required ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }
    if (active_sequence_count != linear_plan->maximum_active_sequence_count)
    {
        return plan_is_required ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
    }
    if (plan_was_launched != 0)
    {
        *plan_was_launched = true;
    }
    return SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan(
        linear_plan,
        input,
        weight,
        output,
        active_sequence_count,
        cuda_stream);
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

    plan_is_required = node_context->projection_backend_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT;
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
        return status;
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
        return status;
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
        return status;
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
        return status;
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
        return status;
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
        return status;
    }
    return SparkGlm52ResidentDecodeStageLaunchRawLinear(
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
    SparkGlm52ResidentDecodeStageMoeSiluKernel<<<
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

static SparkStatus SparkGlm52ResidentDecodeStageTryLaunchDriverGroupedMoe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    SparkGlm52ResidentDecodeStageGroupedMoeLaunchFunction launch_function;
    SparkStatus status;

    if (node_context->grouped_moe_plan == 0 ||
        node_context->grouped_moe_plan->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_ABI_VERSION ||
        node_context->grouped_moe_plan->launch_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    launch_function =
        (SparkGlm52ResidentDecodeStageGroupedMoeLaunchFunction)
            node_context->grouped_moe_plan->launch_function;
    status = launch_function(
        node_context->grouped_moe_plan,
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

static SparkStatus SparkGlm52ResidentDecodeStageLaunchLocalMoe(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream,
    uint32_t active_sequence_count)
{
    uint32_t route_count;
    uint64_t intermediate_value_count;
    uint64_t hidden_element_count;
    dim3 gate_up_grid;
    dim3 down_grid;
    SparkStatus status;

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
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP &&
        node_context->mlp_execution_mode ==
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE)
    {
        return SparkGlm52ResidentDecodeStageLaunchPreboundDenseMlp(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }
    if (node_context->mlp_execution_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_DRIVER_GROUPED_MOE)
    {
        return SparkGlm52ResidentDecodeStageTryLaunchDriverGroupedMoe(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    }
    if (node_context->layer_progression_mode ==
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP)
    {
        gate_up_grid = dim3(
            node_context->dense_intermediate_dimension,
            active_sequence_count,
            1u);
        SparkGlm52ResidentDecodeStageDenseGateUpKernel<<<
            gate_up_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            (const uint16_t *)node_context->dense_gate_weight_bf16,
            (const uint16_t *)node_context->dense_up_weight_bf16,
            (uint16_t *)pipeline_slot->moe_gate_bf16,
            (uint16_t *)pipeline_slot->moe_up_bf16,
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        intermediate_value_count =
            (uint64_t)active_sequence_count *
            (uint64_t)node_context->dense_intermediate_dimension;
        SparkGlm52ResidentDecodeStageMoeSiluKernel<<<
            SparkGlm52ResidentDecodeStageElementBlockCount(
                intermediate_value_count),
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
        down_grid = dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            active_sequence_count,
            1u);
        SparkGlm52ResidentDecodeStageDenseDownResidualKernel<<<
            down_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
            (const uint16_t *)pipeline_slot->moe_intermediate_bf16,
            (const uint16_t *)node_context->dense_down_weight_bf16,
            (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->dense_intermediate_dimension);
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
        SparkGlm52ResidentDecodeStageMoeRouterTopKKernel<<<
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            (const uint16_t *)node_context->moe_router_weight_bf16,
            node_context->moe_router_score_bias_f32,
            pipeline_slot->moe_topk_expert_ids,
            pipeline_slot->moe_topk_weights,
            active_sequence_count,
            node_context->moe_expert_count,
            node_context->moe_top_k,
            node_context->moe_norm_topk_prob,
            node_context->moe_routed_scaling_factor);
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
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOP1)
    {
        uint8_t *hidden_payload_u8;
        uint8_t *hidden_scale_e4m3;
        uint8_t *intermediate_payload_u8;
        uint8_t *intermediate_scale_e4m3;

        SparkGlm52ResidentDecodeStageMoeRouterTopKKernel<<<
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            (const uint16_t *)node_context->moe_router_weight_bf16,
            node_context->moe_router_score_bias_f32,
            pipeline_slot->moe_topk_expert_ids,
            pipeline_slot->moe_topk_weights,
            active_sequence_count,
            node_context->moe_expert_count,
            node_context->moe_top_k,
            node_context->moe_norm_topk_prob,
            node_context->moe_routed_scaling_factor);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        hidden_payload_u8 = (uint8_t *)pipeline_slot->moe_route_output_bf16;
        hidden_scale_e4m3 =
            hidden_payload_u8 +
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
        SparkGlm52ResidentDecodeStageBf16ToNvfp4Kernel<<<
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            32u,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            hidden_payload_u8,
            hidden_scale_e4m3,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            1.0f / node_context->moe_nvfp4_gate_input_scale);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52ResidentDecodeStageMoeNvfp4GateUpTop1Kernel<<<
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            hidden_payload_u8,
            hidden_scale_e4m3,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_gate_weight_u8,
            node_context->moe_nvfp4_gate_weight_scale_e4m3,
            node_context->moe_nvfp4_up_weight_u8,
            node_context->moe_nvfp4_up_weight_scale_e4m3,
            (uint16_t *)pipeline_slot->moe_gate_bf16,
            (uint16_t *)pipeline_slot->moe_up_bf16,
            node_context->moe_nvfp4_selected_expert_id,
            node_context->moe_nvfp4_gate_input_scale *
                node_context->moe_nvfp4_gate_weight_scale_2,
            node_context->moe_nvfp4_up_input_scale *
                node_context->moe_nvfp4_up_weight_scale_2);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        intermediate_payload_u8 = (uint8_t *)pipeline_slot->moe_gate_bf16;
        intermediate_scale_e4m3 =
            intermediate_payload_u8 +
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
             2u);
        SparkGlm52ResidentDecodeStageMoeNvfp4SiluAndQuantTop1Kernel<<<
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            32u,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->moe_gate_bf16,
            (const uint16_t *)pipeline_slot->moe_up_bf16,
            pipeline_slot->moe_topk_expert_ids,
            (uint16_t *)pipeline_slot->moe_intermediate_bf16,
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            node_context->moe_nvfp4_selected_expert_id,
            1.0f / node_context->moe_nvfp4_down_input_scale);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52ResidentDecodeStageMoeNvfp4DownTop1Kernel<<<
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_down_weight_u8,
            node_context->moe_nvfp4_down_weight_scale_e4m3,
            (uint16_t *)pipeline_slot->moe_route_output_bf16,
            node_context->moe_nvfp4_selected_expert_id,
            node_context->moe_nvfp4_down_input_scale *
                node_context->moe_nvfp4_down_weight_scale_2);
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
        SparkGlm52ResidentDecodeStageMoeCombineKernel<<<
            SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
            (const uint16_t *)pipeline_slot->moe_route_output_bf16,
            pipeline_slot->moe_topk_weights,
            (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            1u);
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
        uint32_t nvfp4_route_count;
        uint8_t *hidden_payload_u8;
        uint8_t *hidden_scale_e4m3;
        uint8_t *intermediate_payload_u8;
        uint8_t *intermediate_scale_e4m3;
        dim3 hidden_quant_grid;
        dim3 gate_up_nvfp4_grid;
        dim3 intermediate_quant_grid;
        dim3 down_nvfp4_grid;

        nvfp4_route_count =
            active_sequence_count * node_context->moe_top_k;
        SparkGlm52ResidentDecodeStageMoeRouterTopKKernel<<<
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            (const uint16_t *)node_context->moe_router_weight_bf16,
            node_context->moe_router_score_bias_f32,
            pipeline_slot->moe_topk_expert_ids,
            pipeline_slot->moe_topk_weights,
            active_sequence_count,
            node_context->moe_expert_count,
            node_context->moe_top_k,
            node_context->moe_norm_topk_prob,
            node_context->moe_routed_scaling_factor);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (pipeline_slot->moe_bound_expert_slots != 0)
        {
            SparkGlm52ResidentDecodeStageResolveBoundExpertSlotsKernel<<<
                SparkGlm52ResidentDecodeStageElementBlockCount(
                    nvfp4_route_count),
                SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
                0u,
                cuda_stream>>>(
                pipeline_slot->moe_topk_expert_ids,
                node_context->moe_nvfp4_bound_expert_ids,
                node_context->moe_nvfp4_expert_id_to_bound_slot,
                pipeline_slot->moe_bound_expert_slots,
                nvfp4_route_count,
                node_context->moe_nvfp4_bound_expert_count);
            status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
                node_context,
                cuda_slot_state,
                cuda_stream);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        hidden_payload_u8 = (uint8_t *)pipeline_slot->moe_route_output_bf16;
        hidden_scale_e4m3 =
            hidden_payload_u8 +
            ((uint64_t)nvfp4_route_count *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u));
        hidden_quant_grid = dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            nvfp4_route_count);
        SparkGlm52ResidentDecodeStageBf16ToNvfp4RoutesKernel<<<
            hidden_quant_grid,
            32u,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_bound_expert_ids,
            node_context->moe_nvfp4_expert_id_to_bound_slot,
            pipeline_slot->moe_bound_expert_slots,
            node_context->moe_nvfp4_gate_input_scale_f32,
            hidden_payload_u8,
            hidden_scale_e4m3,
            nvfp4_route_count,
            node_context->moe_nvfp4_bound_expert_count);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        gate_up_nvfp4_grid = dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,
            nvfp4_route_count);
        SparkGlm52ResidentDecodeStageMoeNvfp4GateUpTopKKernel<<<
            gate_up_nvfp4_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            hidden_payload_u8,
            hidden_scale_e4m3,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_bound_expert_ids,
            node_context->moe_nvfp4_expert_id_to_bound_slot,
            pipeline_slot->moe_bound_expert_slots,
            node_context->moe_nvfp4_gate_weight_u8,
            node_context->moe_nvfp4_gate_weight_scale_e4m3,
            node_context->moe_nvfp4_up_weight_u8,
            node_context->moe_nvfp4_up_weight_scale_e4m3,
            node_context->moe_nvfp4_gate_input_scale_f32,
            node_context->moe_nvfp4_gate_weight_scale_2_f32,
            node_context->moe_nvfp4_up_input_scale_f32,
            node_context->moe_nvfp4_up_weight_scale_2_f32,
            (uint16_t *)pipeline_slot->moe_gate_bf16,
            (uint16_t *)pipeline_slot->moe_up_bf16,
            nvfp4_route_count,
            node_context->moe_nvfp4_bound_expert_count);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        intermediate_payload_u8 = (uint8_t *)pipeline_slot->moe_gate_bf16;
        intermediate_scale_e4m3 =
            intermediate_payload_u8 +
            ((uint64_t)nvfp4_route_count *
             (SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
              2u));
        intermediate_quant_grid = dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION /
                SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE,
            nvfp4_route_count);
        SparkGlm52ResidentDecodeStageMoeNvfp4SiluAndQuantTopKKernel<<<
            intermediate_quant_grid,
            32u,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->moe_gate_bf16,
            (const uint16_t *)pipeline_slot->moe_up_bf16,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_bound_expert_ids,
            node_context->moe_nvfp4_expert_id_to_bound_slot,
            pipeline_slot->moe_bound_expert_slots,
            node_context->moe_nvfp4_down_input_scale_f32,
            (uint16_t *)pipeline_slot->moe_intermediate_bf16,
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            nvfp4_route_count,
            node_context->moe_nvfp4_bound_expert_count);
        status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
            node_context,
            cuda_slot_state,
            cuda_stream);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        down_nvfp4_grid = dim3(
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            nvfp4_route_count);
        SparkGlm52ResidentDecodeStageMoeNvfp4DownTopKKernel<<<
            down_nvfp4_grid,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            intermediate_payload_u8,
            intermediate_scale_e4m3,
            pipeline_slot->moe_topk_expert_ids,
            node_context->moe_nvfp4_bound_expert_ids,
            node_context->moe_nvfp4_expert_id_to_bound_slot,
            pipeline_slot->moe_bound_expert_slots,
            node_context->moe_nvfp4_down_weight_u8,
            node_context->moe_nvfp4_down_weight_scale_e4m3,
            node_context->moe_nvfp4_down_input_scale_f32,
            node_context->moe_nvfp4_down_weight_scale_2_f32,
            (uint16_t *)pipeline_slot->moe_route_output_bf16,
            nvfp4_route_count,
            node_context->moe_nvfp4_bound_expert_count);
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
        SparkGlm52ResidentDecodeStageMoeCombineKernel<<<
            SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
            SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
            (const uint16_t *)pipeline_slot->moe_route_output_bf16,
            pipeline_slot->moe_topk_weights,
            (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
            active_sequence_count,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
            node_context->moe_top_k);
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

    route_count = active_sequence_count *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
    gate_up_grid = dim3(
        node_context->moe_intermediate_dimension,
        route_count,
        1u);
    SparkGlm52ResidentDecodeStageMoeGateUpKernel<<<
        gate_up_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_normalized_hidden_bf16,
        pipeline_slot->moe_topk_expert_ids,
        (const uint16_t *)node_context->moe_gate_weight_bf16,
        (const uint16_t *)node_context->moe_up_weight_bf16,
        (uint16_t *)pipeline_slot->moe_gate_bf16,
        (uint16_t *)pipeline_slot->moe_up_bf16,
        route_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->moe_intermediate_dimension,
        node_context->moe_first_bound_expert_id,
        node_context->moe_bound_expert_count);
    status = SparkGlm52ResidentDecodeStageCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    intermediate_value_count =
        (uint64_t)route_count *
        (uint64_t)node_context->moe_intermediate_dimension;
    SparkGlm52ResidentDecodeStageMoeSiluKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(
            intermediate_value_count),
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
    down_grid = dim3(
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        route_count,
        1u);
    SparkGlm52ResidentDecodeStageMoeDownKernel<<<
        down_grid,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->moe_intermediate_bf16,
        pipeline_slot->moe_topk_expert_ids,
        (const uint16_t *)node_context->moe_down_weight_bf16,
        (uint16_t *)pipeline_slot->moe_route_output_bf16,
        route_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->moe_intermediate_dimension,
        node_context->moe_first_bound_expert_id,
        node_context->moe_bound_expert_count);
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
    SparkGlm52ResidentDecodeStageMoeCombineKernel<<<
        SparkGlm52ResidentDecodeStageElementBlockCount(hidden_element_count),
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->post_attention_hidden_bf16,
        (const uint16_t *)pipeline_slot->moe_route_output_bf16,
        pipeline_slot->moe_topk_weights,
        (uint16_t *)pipeline_slot->layer_output_hidden_bf16,
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
        node_context->moe_top_k);
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

extern "C" SparkStatus SparkGlm52ResidentDecodeStageBackendSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state;
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    cudaGraph_t captured_graph;
    cudaGraphExec_t captured_graph_exec;
    uint32_t graph_capture_active;
    dim3 attention_grid;
    const uint16_t *final_norm_input_bf16;
    uint64_t hidden_element_count;
    bool full_stage_plan_was_launched;
    SparkStatus status;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
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
    pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
    cuda_slot_state = node_context->cuda_pipeline_slot_states != 0
        ? &node_context->cuda_pipeline_slot_states[pipeline_slot_index]
        : 0;
    cuda_stream = (cudaStream_t)pipeline_slot->cuda_stream;
    captured_graph = 0;
    captured_graph_exec = 0;
    graph_capture_active = 0u;
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
        cuda_slot_state->graph_active_sequence_count == active_sequence_count)
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
        if (cuda_slot_state->cuda_graph_exec != 0)
        {
            cudaGraphExecDestroy((cudaGraphExec_t)cuda_slot_state->cuda_graph_exec);
            cuda_slot_state->cuda_graph_exec = 0;
            cuda_slot_state->graph_active_sequence_count = 0u;
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

    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED);
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
        return status;
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
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = node_context->projection_mode !=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16
        ? SparkGlm52ResidentDecodeStageLaunchRawGlmProjection(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count)
        : SparkGlm52ResidentDecodeStageLaunchLoweredProjection(
            node_context,
            pipeline_slot,
            cuda_slot_state,
            cuda_stream,
            active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
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
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE);
    if (status != SPARK_STATUS_OK)
    {
        return status;
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
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
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
        return status;
    }
    status = SparkGlm52ResidentDecodeStageMaybeMarkPhase(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52ResidentDecodeStageLaunchLocalMoe(
        node_context,
        pipeline_slot,
        cuda_slot_state,
        cuda_stream,
        active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
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
    SparkGlm52ResidentDecodeStageRestrictedArgmaxKernel<<<
        active_sequence_count,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        0u,
        cuda_stream>>>(
        pipeline_slot->restricted_logits,
        node_context->restricted_token_ids,
        pipeline_slot->restricted_selected_token_ids,
        pipeline_slot->restricted_selected_token_scores,
        active_sequence_count);
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
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_RESTRICTED_LOGITS);
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
    SparkGlm52ResidentDecodeStageMtpArgmaxKernel<<<
        active_sequence_count *
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,
        0u,
        cuda_stream>>>(
        pipeline_slot->mtp_draft_logits,
        node_context->restricted_token_ids,
        pipeline_slot->mtp_draft_token_ids,
        active_sequence_count);
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
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_DRAFT);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ResidentDecodeStageMtpVerifyCommitKernel<<<
        active_sequence_count,
        1u,
        0u,
        cuda_stream>>>(
        pipeline_slot->mtp_target_token_ids,
        pipeline_slot->mtp_draft_token_ids,
        pipeline_slot->mtp_accept_mask,
        pipeline_slot->mtp_committed_token_ids,
        pipeline_slot->mtp_event_counters,
        active_sequence_count);
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
        cuda_slot_state->graph_active_sequence_count = active_sequence_count;
        cuda_slot_state->graph_capture_count += 1u;
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

extern "C" void SparkGlm52ResidentDecodeStageBackendQuiesce(
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
        }
    }
}
