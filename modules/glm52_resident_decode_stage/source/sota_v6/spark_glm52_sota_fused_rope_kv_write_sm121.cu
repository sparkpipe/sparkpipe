#include "spark_glm52_sota_decode_common.cuh"

static __global__ __launch_bounds__(256, 2)
void SparkGlm52SotaFusedRopeKvWriteBf16Kernel(
    SparkGlm52SotaRopeKvArguments arguments)
{
    uint32_t token_index = blockIdx.x;
    uint32_t lane_index = threadIdx.x;
    uint32_t cache_token_index;
    uint32_t position_index;

    if (token_index >= arguments.active_token_count)
    {
        return;
    }

    cache_token_index = arguments.cache_token_indices[token_index];
    position_index = arguments.position_indices[token_index];
    if (cache_token_index >= arguments.cache_token_capacity)
    {
        return;
    }

    for (uint32_t column = lane_index * 2u; column + 1u < SPARK_GLM52_SOTA_LATENT_DIMENSION; column += blockDim.x * 2u)
    {
        __nv_bfloat162 latent_pair = *reinterpret_cast<const __nv_bfloat162 *>(
            &arguments.kv_latent_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_LATENT_DIMENSION) + column]);
        *reinterpret_cast<__nv_bfloat162 *>(
            &arguments.latent_cache_bf16[((uint64_t)cache_token_index * arguments.cache_token_stride_elements) + column]) = latent_pair;
    }

    for (uint32_t rope_pair = lane_index; rope_pair < SPARK_GLM52_SOTA_ROPE_DIMENSION / 2u; rope_pair += blockDim.x)
    {
        uint32_t column = rope_pair * 2u;
        uint64_t table_index = ((uint64_t)position_index * (SPARK_GLM52_SOTA_ROPE_DIMENSION / 2u)) + rope_pair;
        float cosine = __ldg(&arguments.cos_f32[table_index]);
        float sine = __ldg(&arguments.sin_f32[table_index]);
        float query_x = SparkGlm52SotaBf16ToFloat(arguments.query_rope_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column]);
        float query_y = SparkGlm52SotaBf16ToFloat(arguments.query_rope_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column + 1u]);
        float key_x = SparkGlm52SotaBf16ToFloat(arguments.key_rope_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column]);
        float key_y = SparkGlm52SotaBf16ToFloat(arguments.key_rope_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column + 1u]);
        float query_rotated_x = fmaf(query_x, cosine, -query_y * sine);
        float query_rotated_y = fmaf(query_x, sine, query_y * cosine);
        float key_rotated_x = fmaf(key_x, cosine, -key_y * sine);
        float key_rotated_y = fmaf(key_x, sine, key_y * cosine);

        arguments.query_rope_out_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column] = SparkGlm52SotaFloatToBf16(query_rotated_x);
        arguments.query_rope_out_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_ROPE_DIMENSION) + column + 1u] = SparkGlm52SotaFloatToBf16(query_rotated_y);
        arguments.key_rope_cache_bf16[
            ((uint64_t)cache_token_index * arguments.cache_token_stride_elements) +
            SPARK_GLM52_SOTA_LATENT_DIMENSION + column] = SparkGlm52SotaFloatToBf16(key_rotated_x);
        arguments.key_rope_cache_bf16[
            ((uint64_t)cache_token_index * arguments.cache_token_stride_elements) +
            SPARK_GLM52_SOTA_LATENT_DIMENSION + column + 1u] = SparkGlm52SotaFloatToBf16(key_rotated_y);
    }

    for (uint32_t value_index = lane_index * 2u;
         value_index + 1u < SPARK_GLM52_SOTA_HEAD_COUNT * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION;
         value_index += blockDim.x * 2u)
    {
        __nv_bfloat162 value_pair = *reinterpret_cast<const __nv_bfloat162 *>(
            &arguments.value_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION) + value_index]);
        *reinterpret_cast<__nv_bfloat162 *>(
            &arguments.value_cache_bf16[
                ((uint64_t)cache_token_index * arguments.cache_token_stride_elements) +
                SPARK_GLM52_SOTA_LATENT_DIMENSION + SPARK_GLM52_SOTA_ROPE_DIMENSION + value_index]) = value_pair;
    }
}

static __global__ __launch_bounds__(128, 4)
void SparkGlm52SotaQuantizedKvWriteKernel(
    SparkGlm52SotaRopeKvArguments arguments)
{
    uint32_t warp_index = threadIdx.x >> 5u;
    uint32_t lane_index = threadIdx.x & 31u;
    uint32_t token_index = blockIdx.y;
    uint32_t group_index = (blockIdx.x * 4u) + warp_index;
    uint32_t cache_token_index;
    uint32_t base_column;
    float value = 0.0f;
    float group_max;
    float scale;
    uint8_t scale_e4m3;

    if (token_index >= arguments.active_token_count || warp_index >= 4u)
    {
        return;
    }
    cache_token_index = arguments.cache_token_indices[token_index];
    if (cache_token_index >= arguments.cache_token_capacity)
    {
        return;
    }

    if (group_index < (SPARK_GLM52_SOTA_LATENT_DIMENSION / SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE))
    {
        base_column = group_index * SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE;
        if (lane_index < SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE)
        {
            value = SparkGlm52SotaBf16ToFloat(arguments.kv_latent_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_LATENT_DIMENSION) + base_column + lane_index]);
        }
        group_max = SparkGlm52SotaWarpReduceMax(lane_index < SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE ? fabsf(value) : 0.0f);
        group_max = __shfl_sync(0xffffffffu, group_max, 0);
        scale = fmaxf(group_max * 0.1666666667f, 1.0e-8f);
        scale_e4m3 = SparkGlm52SotaEncodePositiveE4m3(scale);
        scale = SparkGlm52SotaDecodeE4m3(scale_e4m3);
        if (lane_index == 0u)
        {
            arguments.latent_cache_nvfp4.scale_e4m3_u8[((uint64_t)cache_token_index * arguments.latent_cache_nvfp4.scale_row_stride_bytes) + group_index] = scale_e4m3;
        }
        if (lane_index < 8u)
        {
            float first_value = SparkGlm52SotaBf16ToFloat(arguments.kv_latent_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_LATENT_DIMENSION) + base_column + (lane_index * 2u)]);
            float second_value = SparkGlm52SotaBf16ToFloat(arguments.kv_latent_in_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_LATENT_DIMENSION) + base_column + (lane_index * 2u) + 1u]);
            uint64_t packed_index = ((uint64_t)cache_token_index * arguments.latent_cache_nvfp4.packed_row_stride_bytes) + ((uint64_t)base_column >> 1u) + lane_index;

            SparkGlm52SotaStoreNvfp4Pair(
                arguments.latent_cache_nvfp4.payload_u8,
                packed_index,
                SparkGlm52SotaEncodeE2m1(first_value / scale),
                SparkGlm52SotaEncodeE2m1(second_value / scale));
        }
    }
}

extern "C" cudaError_t SparkGlm52SotaFusedRopeKvWriteSm121(
    const SparkGlm52SotaRopeKvArguments *arguments,
    cudaStream_t stream)
{
    if (arguments == 0 || arguments->query_rope_in_bf16 == 0 || arguments->key_rope_in_bf16 == 0 ||
        arguments->kv_latent_in_bf16 == 0 || arguments->cos_f32 == 0 || arguments->sin_f32 == 0 ||
        arguments->position_indices == 0 || arguments->cache_token_indices == 0 ||
        arguments->query_rope_out_bf16 == 0 || arguments->latent_cache_bf16 == 0 ||
        arguments->key_rope_cache_bf16 == 0)
    {
        return cudaErrorInvalidValue;
    }

    SparkGlm52SotaFusedRopeKvWriteBf16Kernel<<<arguments->active_token_count, 256u, 0u, stream>>>(*arguments);
    if (arguments->write_nvfp4 != 0u)
    {
        dim3 grid;

        if (arguments->latent_cache_nvfp4.payload_u8 == 0 || arguments->latent_cache_nvfp4.scale_e4m3_u8 == 0)
        {
            return cudaErrorInvalidValue;
        }
        grid.x = SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_LATENT_DIMENSION / SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE, 4u);
        grid.y = arguments->active_token_count;
        grid.z = 1u;
        SparkGlm52SotaQuantizedKvWriteKernel<<<grid, 128u, 0u, stream>>>(*arguments);
    }
    return cudaGetLastError();
}
