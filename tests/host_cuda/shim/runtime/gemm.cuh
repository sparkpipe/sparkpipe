#pragma once

#include <stdint.h>
#include <vector>

#include "inference/kernels/gemm.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "runtime/launch.h"

struct LmRecordedGemm
{
    const void *activation;
    const void *weight;
    const void *output;
    LmScaleTensor activation_scale;
    LmScaleTensor weight_scale;
    uint32_t input_dimension;
    uint32_t output_dimension;
    uint32_t packed_rows;
    bool grouped;
    bool indirect;
};

extern std::vector<LmRecordedGemm> lm_recorded_gemms;

template<
    class FormatA,
    class FormatB,
    uint32_t TILE_N,
    uint32_t TILE_K,
    uint32_t STAGES,
    uint32_t WARPS>
static int32_t LmGemmLaunchAsymmetric(
    LmGemmArguments *args,
    const void *activation_bytes,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    LmRecordedGemm record;
    uint32_t row;
    uint32_t element;

    record.activation = activation_bytes;
    record.weight = weight_bytes;
    record.output = args->output_f32 != 0
        ? args->output_f32
        : args->output_bf16;
    record.activation_scale = args->scale_a;
    record.weight_scale = args->scale_b;
    record.input_dimension = input_dimension;
    record.output_dimension = output_dimension;
    record.packed_rows = packed_rows;
    record.grouped = grouped;
    record.indirect = args->source_row_map != 0;
    lm_recorded_gemms.push_back(record);

    for (row = 0u; row < packed_rows; ++row)
    {
        for (element = 0u; element < output_dimension; ++element)
        {
            uint32_t output_row_stride = args->output_row_stride != 0u
                ? args->output_row_stride
                : output_dimension;
            uint64_t output_index =
                ((uint64_t)row * output_row_stride) +
                args->output_column_offset + element;
            float value = 0.125f * (float)lm_recorded_gemms.size();

            if (args->output_f32 != 0)
            {
                ((float *)args->output_f32)[output_index] = value;
                continue;
            }
            if (args->accumulate_bf16 != 0)
            {
                uint16_t *accumulation =
                    (uint16_t *)args->accumulate_bf16;
                float sum = LmBf16ToFloat(accumulation[output_index]) + value;

                accumulation[output_index] = LmFloatToBf16(sum);
                if (args->accumulate_bf16 == args->output_bf16)
                    continue;
            }
            ((uint16_t *)args->output_bf16)[output_index] =
                LmFloatToBf16(value);
        }
    }

    (void)tokens;
    (void)top_k;
    (void)group_count;
    (void)multiprocessors;
    (void)stream;
    (void)sizeof(FormatA);
    (void)sizeof(FormatB);
    (void)TILE_N;
    (void)TILE_K;
    (void)STAGES;
    (void)WARPS;
    return LM_LAUNCH_OK;
}


template<
    class WeightFormat,
    uint32_t TILE_N,
    uint32_t STAGES,
    uint32_t WARPS>
static int32_t LmGemmWeightOnlyLaunch(
    LmGemmArguments *args,
    const void *activation_bf16,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    return LmGemmLaunchAsymmetric<
        LmBf16Format,
        WeightFormat,
        TILE_N,
        LmBf16Format::kTileK,
        STAGES,
        WARPS>(
            args,
            activation_bf16,
            weight_bytes,
            packed_rows,
            tokens,
            top_k,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            stream);
}

template<
    class WeightFormat,
    uint32_t TILE_N,
    uint32_t STAGES,
    uint32_t WARPS>
static int32_t LmGemmWeightOnlyIndirectLaunch(
    LmGemmArguments *args,
    const void *activation_bf16,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    if (args == 0 || args->source_row_map == 0 ||
        args->source_row_count != tokens)
        return LM_LAUNCH_ERR_SHAPE;
    args->activation_bytes = activation_bf16;
    return LmGemmWeightOnlyLaunch<WeightFormat,TILE_N,STAGES,WARPS>(
        args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,
        group_count,input_dimension,output_dimension,multiprocessors,true,
        stream);
}

// INTERLEAVED_B recorder shims: the pack V2 interleaved launchers record the
// same way as the plain weight-only launchers (the host layer gate only
// inspects dataflow, not the staged cell grid), at the interleaved TILE_K.
template<
    class WeightFormat,
    uint32_t TILE_N,
    uint32_t STAGES,
    uint32_t WARPS,
    uint32_t TILE_K = 128u>
static int32_t LmGemmWeightOnlyInterleavedLaunch(
    LmGemmArguments *args,
    const void *activation_bf16,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    return LmGemmLaunchAsymmetric<
        LmBf16Format,WeightFormat,TILE_N,TILE_K,STAGES,WARPS>(
            args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,
            group_count,input_dimension,output_dimension,multiprocessors,
            grouped,stream);
}

template<
    class WeightFormat,
    uint32_t TILE_N,
    uint32_t STAGES,
    uint32_t WARPS,
    uint32_t TILE_K = 128u>
static int32_t LmGemmWeightOnlyIndirectInterleavedLaunch(
    LmGemmArguments *args,
    const void *activation_bf16,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    if (args == 0 || args->source_row_map == 0 ||
        args->source_row_count != tokens)
        return LM_LAUNCH_ERR_SHAPE;
    args->activation_bytes = activation_bf16;
    return LmGemmWeightOnlyInterleavedLaunch<WeightFormat,TILE_N,STAGES,WARPS,TILE_K>(
        args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,
        group_count,input_dimension,output_dimension,multiprocessors,true,
        stream);
}

template<
    class Format,
    uint32_t TILE_N,
    uint32_t TILE_K,
    uint32_t STAGES,
    uint32_t WARPS>
static int32_t LmGemmLaunch(
    LmGemmArguments *args,
    const void *activation_bytes,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    return LmGemmLaunchAsymmetric<
        Format,
        Format,
        TILE_N,
        TILE_K,
        STAGES,
        WARPS>(
            args,
            activation_bytes,
            weight_bytes,
            packed_rows,
            tokens,
            top_k,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            stream);
}

// GEMM-008 mirror: the shape-based TILE_K fallback records the selected tile
// so host tests can pin the dispatch without a CUDA toolchain.
template<class Format, uint32_t TILE_N, uint32_t STAGES, uint32_t WARPS>
static int32_t LmGemmLaunchTileK(
    LmGemmArguments *args,
    const void *activation_bytes,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    uint32_t tile_k = LmGemmSelectTileK(Format::kTileK, input_dimension);
    if ( tile_k == Format::kTileK )
        return LmGemmLaunch<Format,TILE_N,Format::kTileK,STAGES,WARPS>(
            args,activation_bytes,weight_bytes,packed_rows,tokens,top_k,
            group_count,input_dimension,output_dimension,multiprocessors,
            grouped,stream);
    if ( tile_k == 32u )
        return LmGemmLaunch<Format,TILE_N,32u,STAGES,WARPS>(
            args,activation_bytes,weight_bytes,packed_rows,tokens,top_k,
            group_count,input_dimension,output_dimension,multiprocessors,
            grouped,stream);
    return LM_LAUNCH_ERR_SHAPE;
}
