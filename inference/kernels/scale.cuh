#pragma once

#include "inference/kernels/dtype.cuh"
#include <stdint.h>

#if !defined(__CUDACC__) && !defined(__forceinline__)
#define __forceinline__ inline
#define LM_SCALE_DEFINED_FORCEINLINE 1
#endif

#if defined(__CUDACC__)
#define LM_SCALE_HOST_DEVICE __host__ __device__
#define LM_SCALE_DEVICE __device__
#else
#define LM_SCALE_HOST_DEVICE
#define LM_SCALE_DEVICE
#endif

enum LmScaleEncoding : uint32_t
{
    LM_SCALE_ENCODING_NONE = 0u,
    LM_SCALE_ENCODING_F32 = 1u,
    LM_SCALE_ENCODING_UE4M3 = 2u,
    LM_SCALE_ENCODING_UE8M0 = 3u,
    LM_SCALE_ENCODING_UE4M3_F32_GLOBAL = 4u
};

struct LmScaleTensor
{
    const void *data;
    const void *global_data;
    uint64_t group_stride_entries;
    uint64_t row_group_stride_entries;
    uint32_t group_count;
    uint32_t row_count;
    uint32_t input_dimension;
    uint32_t row_group_size;
    uint32_t k_group_size;
    uint32_t encoding;
    uint32_t global_encoding;
    uint32_t global_group_count;
    uint32_t reserved;
};

static LM_SCALE_HOST_DEVICE __forceinline__ uint64_t LmCeilDivideU64(
    uint64_t value,
    uint64_t divisor)
{
    if (divisor == 0u)
    {
        return 0u;
    }
    return (value / divisor) + ((value % divisor) != 0u ? 1u : 0u);
}

static LM_SCALE_HOST_DEVICE __forceinline__ uint32_t LmScaleEncodingIsKnown(
    uint32_t encoding)
{
    return encoding <= LM_SCALE_ENCODING_UE4M3_F32_GLOBAL ? 1u : 0u;
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorNone(void)
{
    LmScaleTensor scale;

    scale.data = 0;
    scale.global_data = 0;
    scale.group_stride_entries = 0u;
    scale.row_group_stride_entries = 0u;
    scale.group_count = 0u;
    scale.row_count = 0u;
    scale.input_dimension = 0u;
    scale.row_group_size = 0u;
    scale.k_group_size = 0u;
    scale.encoding = LM_SCALE_ENCODING_NONE;
    scale.global_encoding = LM_SCALE_ENCODING_NONE;
    scale.global_group_count = 0u;
    scale.reserved = 0u;
    return scale;
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorInvalid(
    uint32_t encoding)
{
    LmScaleTensor scale = LmScaleTensorNone();

    scale.encoding = encoding;
    scale.reserved = 1u;
    return scale;
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorBuild(
    const void *data,
    uint32_t encoding,
    uint32_t group_count,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t row_group_size,
    uint32_t k_group_size)
{
    LmScaleTensor scale;
    uint64_t row_group_count;
    uint64_t k_group_count;

    if (encoding == LM_SCALE_ENCODING_NONE)
    {
        return LmScaleTensorNone();
    }
    if (LmScaleEncodingIsKnown(encoding) == 0u || data == 0 ||
        group_count == 0u || row_count == 0u || input_dimension == 0u ||
        row_group_size == 0u || k_group_size == 0u)
    {
        return LmScaleTensorInvalid(encoding);
    }

    row_group_count = LmCeilDivideU64(row_count, row_group_size);
    k_group_count = LmCeilDivideU64(input_dimension, k_group_size);
    if (row_group_count == 0u || k_group_count == 0u ||
        row_group_count > (UINT64_MAX / k_group_count))
    {
        return LmScaleTensorInvalid(encoding);
    }

    scale.data = data;
    scale.global_data = 0;
    scale.group_stride_entries = row_group_count * k_group_count;
    scale.row_group_stride_entries = k_group_count;
    scale.group_count = group_count;
    scale.row_count = row_count;
    scale.input_dimension = input_dimension;
    scale.row_group_size = row_group_size;
    scale.k_group_size = k_group_size;
    scale.encoding = encoding;
    scale.global_encoding = LM_SCALE_ENCODING_NONE;
    scale.global_group_count = 0u;
    scale.reserved = 0u;
    return scale;
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorRowsUe4m3(
    const void *data,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t k_group_size)
{
    return LmScaleTensorBuild(
        data,
        LM_SCALE_ENCODING_UE4M3,
        1u,
        row_count,
        input_dimension,
        1u,
        k_group_size);
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorBlockUe4m3F32Global(
    const void *block_data,
    const void *global_data,
    uint32_t group_count,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t k_group_size)
{
    LmScaleTensor scale = LmScaleTensorBuild(
        block_data,
        LM_SCALE_ENCODING_UE4M3_F32_GLOBAL,
        group_count,
        row_count,
        input_dimension,
        1u,
        k_group_size);
    if (scale.reserved == 0u && global_data != 0)
    {
        scale.global_data = global_data;
        scale.global_encoding = LM_SCALE_ENCODING_F32;
        scale.global_group_count = group_count;
    }
    else
        scale.reserved = 1u;
    return scale;
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorRowsUe8m0(
    const void *data,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t k_group_size)
{
    return LmScaleTensorBuild(
        data,
        LM_SCALE_ENCODING_UE8M0,
        1u,
        row_count,
        input_dimension,
        1u,
        k_group_size);
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorBlockF32(
    const void *data,
    uint32_t group_count,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t row_group_size,
    uint32_t k_group_size)
{
    return LmScaleTensorBuild(
        data,
        LM_SCALE_ENCODING_F32,
        group_count,
        row_count,
        input_dimension,
        row_group_size,
        k_group_size);
}

static LM_SCALE_HOST_DEVICE __forceinline__ LmScaleTensor LmScaleTensorBlockUe8m0(
    const void *data,
    uint32_t group_count,
    uint32_t row_count,
    uint32_t input_dimension,
    uint32_t row_group_size,
    uint32_t k_group_size)
{
    return LmScaleTensorBuild(
        data,
        LM_SCALE_ENCODING_UE8M0,
        group_count,
        row_count,
        input_dimension,
        row_group_size,
        k_group_size);
}

static LM_SCALE_HOST_DEVICE __forceinline__ uint32_t LmScaleTensorIsValid(
    const LmScaleTensor *scale)
{
    uint64_t row_group_count;
    uint64_t k_group_count;

    if (scale == 0 || scale->reserved != 0u ||
        LmScaleEncodingIsKnown(scale->encoding) == 0u)
    {
        return 0u;
    }
    if (scale->encoding == LM_SCALE_ENCODING_NONE)
    {
        return scale->data == 0 &&
            scale->global_data == 0 &&
            scale->group_stride_entries == 0u &&
            scale->row_group_stride_entries == 0u &&
            scale->group_count == 0u &&
            scale->row_count == 0u &&
            scale->input_dimension == 0u &&
            scale->row_group_size == 0u &&
            scale->k_group_size == 0u &&
            scale->global_encoding == LM_SCALE_ENCODING_NONE &&
            scale->global_group_count == 0u;
    }
    if (scale->data == 0 || scale->group_count == 0u ||
        scale->row_count == 0u || scale->input_dimension == 0u ||
        scale->row_group_size == 0u || scale->k_group_size == 0u)
    {
        return 0u;
    }
    if (scale->encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL)
    {
        if (scale->global_data == 0 ||
            scale->global_encoding != LM_SCALE_ENCODING_F32 ||
            scale->global_group_count != scale->group_count)
            return 0u;
    }
    else if (scale->global_data != 0 ||
        scale->global_encoding != LM_SCALE_ENCODING_NONE ||
        scale->global_group_count != 0u)
        return 0u;

    row_group_count = LmCeilDivideU64(
        scale->row_count,
        scale->row_group_size);
    k_group_count = LmCeilDivideU64(
        scale->input_dimension,
        scale->k_group_size);
    if (row_group_count == 0u || k_group_count == 0u ||
        row_group_count > (UINT64_MAX / k_group_count))
    {
        return 0u;
    }
    return scale->row_group_stride_entries == k_group_count &&
        scale->group_stride_entries == row_group_count * k_group_count;
}

static LM_SCALE_HOST_DEVICE __forceinline__ uint32_t LmScaleTensorContains(
    const LmScaleTensor *scale,
    uint32_t group_index,
    uint32_t row_index,
    uint32_t k_index)
{
    if (scale == 0 || scale->encoding == LM_SCALE_ENCODING_NONE)
    {
        return 1u;
    }
    return group_index < scale->group_count &&
        row_index < scale->row_count &&
        k_index < scale->input_dimension;
}

static LM_SCALE_HOST_DEVICE __forceinline__ uint64_t LmScaleTensorIndex(
    const LmScaleTensor *scale,
    uint32_t group_index,
    uint32_t row_index,
    uint32_t k_index)
{
    return ((uint64_t)group_index * scale->group_stride_entries) +
        ((uint64_t)(row_index / scale->row_group_size) *
            scale->row_group_stride_entries) +
        (uint64_t)(k_index / scale->k_group_size);
}

static LM_SCALE_DEVICE __forceinline__ float LmScaleTensorLoadByIndex(
    const LmScaleTensor *scale,
    uint64_t index)
{
    if (scale->encoding == LM_SCALE_ENCODING_NONE)
    {
        return 1.0f;
    }
    if (scale->encoding == LM_SCALE_ENCODING_F32)
    {
        return ((const float *)scale->data)[index];
    }
    if (scale->encoding == LM_SCALE_ENCODING_UE4M3 ||
        scale->encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL)
    {
        return LmUe4m3ToFloat(((const uint8_t *)scale->data)[index]);
    }
    if (scale->encoding == LM_SCALE_ENCODING_UE8M0)
    {
        return LmUe8m0ToFloat(((const uint8_t *)scale->data)[index]);
    }
    return 0.0f;
}

static LM_SCALE_DEVICE __forceinline__ float LmScaleTensorLoad(
    const LmScaleTensor *scale,
    uint32_t group_index,
    uint32_t row_index,
    uint32_t k_index)
{
    if (scale->encoding == LM_SCALE_ENCODING_NONE)
    {
        return 1.0f;
    }
    if (LmScaleTensorContains(scale, group_index, row_index, k_index) == 0u)
    {
        return 1.0f;
    }
    float value = LmScaleTensorLoadByIndex(
        scale,
        LmScaleTensorIndex(scale, group_index, row_index, k_index));
    if (scale->encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL)
        value *= ((const float *)scale->global_data)[group_index];
    return value;
}

#undef LM_SCALE_HOST_DEVICE
#undef LM_SCALE_DEVICE
#ifdef LM_SCALE_DEFINED_FORCEINLINE
#undef __forceinline__
#undef LM_SCALE_DEFINED_FORCEINLINE
#endif
