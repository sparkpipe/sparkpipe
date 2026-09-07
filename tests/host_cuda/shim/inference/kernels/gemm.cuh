#pragma once

#include "inference/kernels/scale.cuh"
#include <stdint.h>

struct LmGemmArguments
{
    LmScaleTensor scale_a;
    LmScaleTensor scale_b;
    uint32_t prefix_built;
    const uint32_t *group_row_offset;
    uint32_t *group_tile_prefix;
    const void *activation_bytes;
    const uint32_t *source_row_map;
    uint32_t source_row_count;
    void *output_bf16;
    void *output_f32;
    void *accumulate_bf16;
    uint32_t output_row_stride;
    uint32_t output_column_offset;
    uint32_t group_count;
    uint32_t input_dimension;
    uint32_t output_dimension;
    void *frame_error;
};
