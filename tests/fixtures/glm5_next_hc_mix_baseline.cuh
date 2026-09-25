#pragma once
#define GLM5_NEXT_HC_MIX_TILE 4096u
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

__global__ void Glm5NextHcSinkhornBaselineKernel(
    const float *__restrict__ mixes_f32,
    const float *__restrict__ scale3_f32,
    const float *__restrict__ base_f32,
    uint32_t row_count,
    uint32_t hc,
    uint32_t iterations,
    float epsilon,
    float *__restrict__ pre_f32,
    float *__restrict__ post_f32,
    float *__restrict__ comb_f32)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i, j, iteration;
    const uint32_t mix_rows = (2u + hc) * hc;
    const float *mixes;
    float comb[16], maximum, total;
    if (row >= row_count || hc > 4u)
        return;
    mixes = mixes_f32 + (uint64_t)row * mix_rows;
    for (i = 0u; i < hc; ++i)
    {
        pre_f32[(uint64_t)row * hc + i] =
            1.0f / (1.0f + __expf(-(mixes[i] * scale3_f32[0] + base_f32[i]))) +
            epsilon;
        post_f32[(uint64_t)row * hc + i] =
            2.0f / (1.0f + __expf(-(mixes[hc + i] * scale3_f32[1] +
                                    base_f32[hc + i])));
    }
    for (i = 0u; i < hc; ++i)
    {
        maximum = -3.0e38f;
        for (j = 0u; j < hc; ++j)
        {
            comb[i * hc + j] =
                mixes[2u * hc + i * hc + j] * scale3_f32[2] +
                base_f32[2u * hc + i * hc + j];
            maximum = fmaxf(maximum, comb[i * hc + j]);
        }
        total = 0.0f;
        for (j = 0u; j < hc; ++j)
            total += (comb[i * hc + j] = __expf(comb[i * hc + j] - maximum));
        for (j = 0u; j < hc; ++j)
            comb[i * hc + j] = comb[i * hc + j] / total + epsilon;
    }
    for (iteration = 0u; iteration < iterations; ++iteration)
    {
        if (iteration != 0u)
            for (i = 0u; i < hc; ++i)
            {
                total = 0.0f;
                for (j = 0u; j < hc; ++j)
                    total += comb[i * hc + j];
                for (j = 0u; j < hc; ++j)
                    comb[i * hc + j] /= total + epsilon;
            }
        for (j = 0u; j < hc; ++j)
        {
            total = 0.0f;
            for (i = 0u; i < hc; ++i)
                total += comb[i * hc + j];
            for (i = 0u; i < hc; ++i)
                comb[i * hc + j] /= total + epsilon;
        }
    }
    for (i = 0u; i < hc * hc; ++i)
        comb_f32[(uint64_t)row * hc * hc + i] = comb[i];
}

__global__ void Glm5NextHcPreReduceBaselineKernel(
    const uint16_t *__restrict__ streams_bf16,
    const float *__restrict__ pre_f32,
    uint16_t *__restrict__ collapsed_bf16,
    uint16_t *__restrict__ snapshot_bf16,
    uint32_t row_count,
    uint32_t hc,
    uint32_t dimension)
{
    uint32_t row = blockIdx.x;
    uint32_t element, stream;
    float value;
    if (row >= row_count)
        return;
    for (element = threadIdx.x; element < dimension;
         element += blockDim.x)
    {
        value = 0.0f;
        for (stream = 0u; stream < hc; ++stream)
        {
            uint64_t index =
                (((uint64_t)row * hc) + stream) * dimension + element;
            uint16_t raw = streams_bf16[index];
            snapshot_bf16[index] = raw;
            value += pre_f32[((uint64_t)row * hc) + stream] *
                LmBf16ToFloat(raw);
        }
        collapsed_bf16[(uint64_t)row * dimension + element] =
            LmFloatToBf16(value);
    }
}
