#pragma once
__global__ void Glm5NextHcMixBaselineKernel(
    const uint16_t *__restrict__ streams_bf16,
    const float *__restrict__ fn_f32,
    float *__restrict__ mixes_f32,
    uint32_t row_count,
    uint32_t flat_dimension,
    uint32_t mix_rows,
    float rms_epsilon)
{
    extern __shared__ float staged[];
    __shared__ float reduction[GLM5_NEXT_LAYER_THREADS / LM_WARP_LANES];
    uint32_t row = blockIdx.x;
    uint32_t warp = threadIdx.x / LM_WARP_LANES;
    uint32_t lane = threadIdx.x % LM_WARP_LANES;
    uint32_t mix, element, tile, tile_end, tile_elements;
    const uint32_t warps = GLM5_NEXT_LAYER_THREADS / LM_WARP_LANES;
    float value, total = 0.0f, accumulator;
    float accum[4];
    if (row >= row_count)
        return;
    for (mix = 0u; mix < 3u; mix++)
        accum[mix] = 0.0f;
    for (tile = 0u; tile < flat_dimension; tile += GLM5_NEXT_HC_MIX_TILE)
    {
        tile_end = tile + GLM5_NEXT_HC_MIX_TILE < flat_dimension
            ? tile + GLM5_NEXT_HC_MIX_TILE
            : flat_dimension;
        tile_elements = tile_end - tile;
        __syncthreads();
        for (element = threadIdx.x; element < tile_elements;
             element += GLM5_NEXT_LAYER_THREADS)
        {
            value = LmBf16ToFloat(
                streams_bf16[((uint64_t)row * flat_dimension) + tile + element]);
            staged[element] = value;
            total += value * value;
        }
        __syncthreads();
        for (mix = warp; mix < mix_rows; mix += warps)
        {
            accumulator = 0.0f;
            for (element = lane; element < tile_elements; element += LM_WARP_LANES)
                accumulator += staged[element] *
                    fn_f32[((uint64_t)mix * flat_dimension) + tile + element];
            for (uint32_t lane_step = LM_WARP_LANES / 2u; lane_step > 0u;
                 lane_step >>= 1)
                accumulator +=
                    __shfl_down_sync(0xFFFFFFFFu, accumulator, lane_step);
            if (lane == 0u)
                accum[mix / warps] += accumulator;
        }
    }
    total = LmBlockSum<GLM5_NEXT_LAYER_THREADS>(total, reduction);
    __shared__ float inverse_shared[1];
    if (threadIdx.x == 0u)
        inverse_shared[0] =
            rsqrtf(total / (float)flat_dimension + rms_epsilon);
    __syncthreads();
    for (mix = warp; mix < mix_rows; mix += warps)
        if (lane == 0u)
            mixes_f32[((uint64_t)row * mix_rows) + mix] =
                accum[mix / warps] * inverse_shared[0];
}
