#include <cuda_runtime.h>
#include <stdint.h>

#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh"

#define SPARK_TEST_EXPORT extern "C" __attribute__((visibility("default")))

SPARK_TEST_EXPORT int dsv41_fp8_linear(uintptr_t weight, uintptr_t scale, uintptr_t input, uintptr_t scratch, uintptr_t output, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension)
{
    return (int)SparkDsv41FlashLaunchFp8BlockLinear(0, (const uint8_t *)weight, (const uint8_t *)scale, (const uint16_t *)input, (float *)scratch, (uint16_t *)output, rows, input_dimension, output_dimension);
}

SPARK_TEST_EXPORT int dsv41_mxfp4_linear(uintptr_t payload, uintptr_t scale, uintptr_t input, uintptr_t scratch, uintptr_t output, uint32_t input_dimension, uint32_t output_dimension)
{
    return (int)SparkDsv41FlashLaunchMxfp4Linear(0, (const uint8_t *)payload, (const uint8_t *)scale, (const uint16_t *)input, (float *)scratch, (uint16_t *)output, input_dimension, output_dimension);
}

SPARK_TEST_EXPORT int dsv41_bf16_linear(uintptr_t weight, uintptr_t input, uintptr_t output_f32, uintptr_t output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
    return (int)SparkDsv41FlashLaunchBf16Linear(0, (const uint16_t *)weight, (const uint16_t *)input, (float *)output_f32, (uint16_t *)output_bf16, input_dimension, output_dimension);
}

SPARK_TEST_EXPORT int dsv41_grouped_fp8_linear(uintptr_t weight, uintptr_t scale, uintptr_t input, uintptr_t output, uint32_t input_dimension, uint32_t output_dimension, uint32_t outputs_per_group)
{
    return (int)SparkDsv41FlashLaunchGroupedFp8Linear(0, (const uint8_t *)weight, (const uint8_t *)scale, (const uint16_t *)input, (uint16_t *)output, input_dimension, output_dimension, outputs_per_group);
}

SPARK_TEST_EXPORT int dsv41_rmsnorm(uintptr_t input, uintptr_t weight, uintptr_t output, uint32_t rows, uint32_t dimension, float epsilon)
{
    return (int)SparkDsv41FlashLaunchRmsNorm(0, (const uint16_t *)input, (const uint16_t *)weight, (uint16_t *)output, rows, dimension, epsilon);
}

SPARK_TEST_EXPORT int dsv41_rope(uintptr_t data, uintptr_t cos_sin, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, uint32_t inverse)
{
    return (int)SparkDsv41FlashLaunchRope(0, (uint16_t *)data, (const float *)cos_sin, heads, head_dimension, rope_dimension, inverse);
}

SPARK_TEST_EXPORT int dsv41_fp8_qdq(uintptr_t data, uint64_t count)
{
    return (int)SparkDsv41FlashLaunchFp8Qdq(0, (uint16_t *)data, count);
}

SPARK_TEST_EXPORT int dsv41_kv_fp4_qdq(uintptr_t data, uint64_t row_stride, uint32_t rows, uint32_t width)
{
    return (int)SparkDsv41FlashLaunchKvFp4Qdq(0, (uint16_t *)data, row_stride, rows, width);
}

SPARK_TEST_EXPORT int dsv41_fp4_pow2_qdq(uintptr_t data, uint32_t groups, uint32_t group_size)
{
    if (data == 0 || groups == 0u || group_size == 0u)
        return (int)cudaErrorInvalidValue;
    SparkDsv41FlashFp4Pow2QdqKernel<<<(groups + 255u) / 256u, 256u>>>((uint16_t *)data, groups, group_size);
    return (int)cudaGetLastError();
}

SPARK_TEST_EXPORT int dsv41_sink_attention(uintptr_t query, uintptr_t keys, uintptr_t indices, uint32_t index_count, uintptr_t sink, uintptr_t output, uint32_t heads, uint32_t head_dimension, float scale)
{
    return (int)SparkDsv41FlashLaunchSinkAttention(0, (const uint16_t *)query, (const uint16_t *)keys, (const int32_t *)indices, index_count, (const float *)sink, (uint16_t *)output, heads, head_dimension, scale);
}

SPARK_TEST_EXPORT int dsv41_gate(uintptr_t input, uintptr_t weight, uintptr_t bias, uintptr_t indices, uintptr_t weights, uint32_t experts, uint32_t dimension, uint32_t topk, float route_scale)
{
    return (int)SparkDsv41FlashLaunchGate(0, (const uint16_t *)input, (const uint16_t *)weight, (const float *)bias, (uint32_t *)indices, (float *)weights, experts, dimension, topk, route_scale);
}

SPARK_TEST_EXPORT int dsv41_swiglu(uintptr_t gate, uintptr_t up, uintptr_t output, uint32_t width, float limit, float weight)
{
    return (int)SparkDsv41FlashLaunchSwiglu(0, (const uint16_t *)gate, (const uint16_t *)up, (uint16_t *)output, width, limit, weight);
}

SPARK_TEST_EXPORT int dsv41_hc_mixes(uintptr_t streams, uintptr_t fn, uintptr_t scale, uintptr_t base, uintptr_t mixes, uintptr_t pre, uintptr_t post, uintptr_t comb, uint32_t hc, uint32_t dimension, uint32_t iterations, float norm_epsilon, float hc_epsilon)
{
    return (int)SparkDsv41FlashLaunchHcMixes(0, (const uint16_t *)streams, (const float *)fn, (const float *)scale, (const float *)base, (float *)mixes, (float *)pre, (float *)post, (float *)comb, hc, dimension, iterations, norm_epsilon, hc_epsilon);
}

SPARK_TEST_EXPORT int dsv41_hc_pre(uintptr_t streams, uintptr_t pre, uintptr_t output, uint32_t hc, uint32_t dimension)
{
    SparkDsv41FlashHcPreKernel<<<(dimension + 255u) / 256u, 256u>>>((const uint16_t *)streams, (const float *)pre, (uint16_t *)output, hc, dimension);
    return (int)cudaGetLastError();
}

SPARK_TEST_EXPORT int dsv41_hc_post(uintptr_t sublayer, uintptr_t residual, uintptr_t post, uintptr_t comb, uintptr_t output, uint32_t hc, uint32_t dimension)
{
    SparkDsv41FlashHcPostKernel<<<(dimension + 255u) / 256u, 256u>>>((const uint16_t *)sublayer, (const uint16_t *)residual, (const float *)post, (const float *)comb, (uint16_t *)output, hc, dimension);
    return (int)cudaGetLastError();
}

SPARK_TEST_EXPORT int dsv41_compress_pool(uintptr_t kv_state, uintptr_t score_state, uintptr_t output, uint32_t ratio, uint32_t width)
{
    SparkDsv41FlashCompressPoolKernel<<<(width + 255u) / 256u, 256u>>>((const float *)kv_state, (const float *)score_state, (uint16_t *)output, ratio, width);
    return (int)cudaGetLastError();
}

SPARK_TEST_EXPORT int dsv41_engram_gate(uintptr_t streams, uintptr_t kv, uintptr_t q_weight, uintptr_t k_weight, uint32_t rows, uint32_t hc, uint32_t dimension, float epsilon)
{
    return (int)SparkDsv41FlashLaunchEngramGate(0, (uint16_t *)streams, (const uint16_t *)kv, (const uint16_t *)q_weight, (const uint16_t *)k_weight, rows, hc, dimension, epsilon);
}

SPARK_TEST_EXPORT int dsv41_candidate_mask(uintptr_t scores, uintptr_t widths, uint64_t row_stride, uintptr_t blocks, uint64_t block_stride, uint32_t rows, uint32_t block_size, uint32_t topk_blocks)
{
    return (int)SparkDsv41FlashLaunchCandidateMask(0, (float *)scores, (const uint32_t *)widths, row_stride, (float *)blocks, block_stride, rows, block_size, topk_blocks);
}

SPARK_TEST_EXPORT int dsv41_index_score(uintptr_t query, uintptr_t keys, uintptr_t head_weights, uintptr_t scores, uint32_t key_count, uint32_t heads, uint32_t head_dimension)
{
    return (int)SparkDsv41FlashLaunchIndexScore(0, (const uint16_t *)query, (const uint16_t *)keys, (const uint16_t *)head_weights, (float *)scores, key_count, heads, head_dimension);
}
