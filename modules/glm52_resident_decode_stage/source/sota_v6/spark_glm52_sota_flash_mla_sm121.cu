#include "spark_glm52_sota_production_plan_sm121.cuh"

#define SPARK_GLM52_FLASH_MLA_TILE_TOKENS 128u
#define SPARK_GLM52_FLASH_MLA_VALUE_TILE 16u

struct SparkGlm52SotaFlashMlaPlanSm121
{
    uint32_t abi_version;
    uint32_t active_token_count;
    uint32_t selected_token_count;
    uint32_t cache_token_stride_elements;
    uint64_t capability_flags;
    float qk_scale;
    const __nv_bfloat16 *query_latent_bf16;
    const __nv_bfloat16 *query_rope_bf16;
    const __nv_bfloat16 *latent_cache_bf16;
    const __nv_bfloat16 *key_rope_cache_bf16;
    const __nv_bfloat16 *value_cache_bf16;
    const uint32_t *sparse_token_indices;
    float *partial_max_f32;
    float *partial_sum_f32;
    float *partial_value_f32;
    __nv_bfloat16 *attention_output_bf16;
};

static __device__ __forceinline__ float SparkGlm52MlaDotLatent512(const __nv_bfloat16 *query, const __nv_bfloat16 *key, uint32_t lane)
{
    float value = 0.0f;
    for (uint32_t index = lane; index < SPARK_GLM52_SOTA_LATENT_DIMENSION; index += 32u)
    {
        value = fmaf(SparkGlm52SotaBf16ToFloat(query[index]), SparkGlm52SotaBf16ToFloat(key[index]), value);
    }
    return SparkGlm52SotaWarpReduceSum(value);
}

static __device__ __forceinline__ float SparkGlm52MlaDotRope64(const __nv_bfloat16 *query, const __nv_bfloat16 *key, uint32_t lane)
{
    float value = 0.0f;
    if (lane < SPARK_GLM52_SOTA_ROPE_DIMENSION)
    {
        value = SparkGlm52SotaBf16ToFloat(query[lane]) * SparkGlm52SotaBf16ToFloat(key[lane]);
    }
    return SparkGlm52SotaWarpReduceSum(value);
}

static __global__ __launch_bounds__(32, 8)
void SparkGlm52FlashMlaTilePassKernel(SparkGlm52SotaFlashMlaPlanSm121 plan)
{
    uint32_t token_index = blockIdx.x;
    uint32_t head_index = blockIdx.y;
    uint32_t tile_count = SparkGlm52SotaCeilDivU32(plan.selected_token_count, SPARK_GLM52_FLASH_MLA_TILE_TOKENS);
    uint32_t tile_index = blockIdx.z % tile_count;
    uint32_t value_tile_index = blockIdx.z / tile_count;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tile_base = tile_index * SPARK_GLM52_FLASH_MLA_TILE_TOKENS;
    uint32_t tile_limit = min(tile_base + SPARK_GLM52_FLASH_MLA_TILE_TOKENS, plan.selected_token_count);
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float partial_values[SPARK_GLM52_FLASH_MLA_VALUE_TILE];
    
    #pragma unroll
    for (uint32_t index = 0u; index < SPARK_GLM52_FLASH_MLA_VALUE_TILE; ++index)
    {
        partial_values[index] = 0.0f;
    }
    if (token_index >= plan.active_token_count || head_index >= SPARK_GLM52_SOTA_HEAD_COUNT)
    {
        return;
    }

    for (uint32_t selected_index = tile_base; selected_index < tile_limit; ++selected_index)
    {
        uint32_t cache_token = plan.sparse_token_indices[((uint64_t)token_index * plan.selected_token_count) + selected_index];
        const __nv_bfloat16 *query_latent = plan.query_latent_bf16 + ((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * SPARK_GLM52_SOTA_LATENT_DIMENSION;
        const __nv_bfloat16 *query_rope = plan.query_rope_bf16 + ((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * SPARK_GLM52_SOTA_ROPE_DIMENSION;
        const __nv_bfloat16 *key_latent = plan.latent_cache_bf16 + ((uint64_t)cache_token * plan.cache_token_stride_elements) + head_index * SPARK_GLM52_SOTA_LATENT_DIMENSION;
        const __nv_bfloat16 *key_rope = plan.key_rope_cache_bf16 + ((uint64_t)cache_token * SPARK_GLM52_SOTA_ROPE_DIMENSION);
        float latent_score = SparkGlm52MlaDotLatent512(query_latent, key_latent, lane);
        float rope_score = SparkGlm52MlaDotRope64(query_rope, key_rope, lane);
        float score = (latent_score + rope_score) * plan.qk_scale;
        float new_max = fmaxf(running_max, score);
        float old_scale = __expf(running_max - new_max);
        float probability_scale = __expf(score - new_max);
        running_sum = running_sum * old_scale + probability_scale;
        #pragma unroll
        for (uint32_t value_offset = 0u; value_offset < SPARK_GLM52_FLASH_MLA_VALUE_TILE; ++value_offset)
        {
            uint32_t value_index = (value_tile_index * SPARK_GLM52_FLASH_MLA_VALUE_TILE) + value_offset;
            float v = value_index < SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION ? SparkGlm52SotaBf16ToFloat(plan.value_cache_bf16[((uint64_t)cache_token * plan.cache_token_stride_elements) + head_index * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION + value_index]) : 0.0f;
            partial_values[value_offset] = partial_values[value_offset] * old_scale + probability_scale * v;
        }
        running_max = new_max;
    }

    if (lane == 0u)
    {
        uint64_t base = (((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * tile_count) + tile_index;
        plan.partial_max_f32[base] = running_max;
        plan.partial_sum_f32[base] = running_sum;
        #pragma unroll
        for (uint32_t value_offset = 0u; value_offset < SPARK_GLM52_FLASH_MLA_VALUE_TILE; ++value_offset)
        {
            uint32_t value_index = (value_tile_index * SPARK_GLM52_FLASH_MLA_VALUE_TILE) + value_offset;
            if (value_index < SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION)
            {
                plan.partial_value_f32[(base * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION) + value_index] = partial_values[value_offset];
            }
        }
    }
}

static __global__ __launch_bounds__(256, 1)
void SparkGlm52FlashMlaFinalReduceKernel(SparkGlm52SotaFlashMlaPlanSm121 plan)
{
    uint32_t token_index = blockIdx.x;
    uint32_t head_index = blockIdx.y;
    uint32_t value_index = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t tile_count = SparkGlm52SotaCeilDivU32(plan.selected_token_count, SPARK_GLM52_FLASH_MLA_TILE_TOKENS);
    float global_max = -CUDART_INF_F;
    float denominator = 0.0f;
    float numerator = 0.0f;

    if (token_index >= plan.active_token_count || head_index >= SPARK_GLM52_SOTA_HEAD_COUNT || value_index >= SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION)
    {
        return;
    }
    for (uint32_t tile_index = 0u; tile_index < tile_count; ++tile_index)
    {
        uint64_t base = (((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * tile_count) + tile_index;
        global_max = fmaxf(global_max, plan.partial_max_f32[base]);
    }
    for (uint32_t tile_index = 0u; tile_index < tile_count; ++tile_index)
    {
        uint64_t base = (((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * tile_count) + tile_index;
        float scale = __expf(plan.partial_max_f32[base] - global_max);
        denominator += plan.partial_sum_f32[base] * scale;
        numerator += plan.partial_value_f32[(base * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION) + value_index] * scale;
    }
    plan.attention_output_bf16[((uint64_t)token_index * SPARK_GLM52_SOTA_HEAD_COUNT + head_index) * SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION + value_index] = SparkGlm52SotaFloatToBf16(numerator / fmaxf(denominator, 1.0e-20f));
}

extern "C" cudaError_t SparkGlm52SotaLaunchFlashMlaSm121(SparkGlm52SotaFlashMlaPlanSm121 *plan, cudaStream_t stream)
{
    uint32_t tile_count;
    uint32_t value_tile_count;
    if (plan == 0 || plan->abi_version != SPARK_GLM52_SOTA_PRODUCTION_PLAN_ABI || plan->active_token_count == 0u || plan->selected_token_count == 0u || plan->selected_token_count > SPARK_GLM52_SOTA_SELECTED_TOKEN_COUNT || plan->query_latent_bf16 == 0 || plan->query_rope_bf16 == 0 || plan->latent_cache_bf16 == 0 || plan->key_rope_cache_bf16 == 0 || plan->value_cache_bf16 == 0 || plan->sparse_token_indices == 0 || plan->partial_max_f32 == 0 || plan->partial_sum_f32 == 0 || plan->partial_value_f32 == 0 || plan->attention_output_bf16 == 0)
    {
        return cudaErrorInvalidValue;
    }
    if ((plan->capability_flags & SPARK_GLM52_SOTA_FAST_CAP_FLASH_MLA) == 0u)
    {
        return cudaErrorInvalidValue;
    }
    tile_count = SparkGlm52SotaCeilDivU32(plan->selected_token_count, SPARK_GLM52_FLASH_MLA_TILE_TOKENS);
    value_tile_count = SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION, SPARK_GLM52_FLASH_MLA_VALUE_TILE);
    SparkGlm52FlashMlaTilePassKernel<<<dim3(plan->active_token_count, SPARK_GLM52_SOTA_HEAD_COUNT, tile_count * value_tile_count), 32u, 0u, stream>>>(*plan);
    SparkGlm52FlashMlaFinalReduceKernel<<<dim3(plan->active_token_count, SPARK_GLM52_SOTA_HEAD_COUNT, SparkGlm52SotaCeilDivU32(SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION, 256u)), 256u, 0u, stream>>>(*plan);
    return cudaGetLastError();
}
