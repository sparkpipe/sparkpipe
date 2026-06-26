#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_mla_kv_kernels.h"

#define SPARK_CUDA_MLA_KV_THREADS 256u

static __device__ float SparkCudaMlaKvBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaMlaKvFloatToBf16(float value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static uint64_t SparkCudaMlaKvQueryPairCountHost(const SparkCudaMlaKvRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->query_head_count * ((uint64_t)request->rope_dim / 2u);
}

static uint64_t SparkCudaMlaKvCacheWorkCountHost(const SparkCudaMlaKvRequest *request)
{
    return (uint64_t)request->token_count * ((uint64_t)request->latent_dim + ((uint64_t)request->rope_dim / 2u));
}

static uint64_t SparkCudaMlaKvCacheEntryCountHost(const SparkCudaMlaKvRequest *request)
{
    return (uint64_t)request->latent_dim + (uint64_t)request->rope_dim;
}

static uint32_t SparkCudaMlaKvBlockCountHost(uint64_t element_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((element_count + (uint64_t)SPARK_CUDA_MLA_KV_THREADS - 1u) / (uint64_t)SPARK_CUDA_MLA_KV_THREADS);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return block_count;
}

static void SparkCudaMlaKvFillReportShape(const SparkCudaMlaKvRequest *request, SparkCudaMlaKvReport *report)
{
    report->query_pair_count = SparkCudaMlaKvQueryPairCountHost(request);
    report->cache_work_count = SparkCudaMlaKvCacheWorkCountHost(request);
    report->cache_entry_count = SparkCudaMlaKvCacheEntryCountHost(request);
}

static __device__ void SparkCudaMlaKvApplyRopePair(uint16_t x0_bf16, uint16_t x1_bf16, float cos_value, float sin_value, uint16_t *out0_bf16, uint16_t *out1_bf16)
{
    float x0;
    float x1;

    x0 = SparkCudaMlaKvBf16ToFloat(x0_bf16);
    x1 = SparkCudaMlaKvBf16ToFloat(x1_bf16);
    *out0_bf16 = SparkCudaMlaKvFloatToBf16((x0 * cos_value) - (x1 * sin_value));
    *out1_bf16 = SparkCudaMlaKvFloatToBf16((x0 * sin_value) + (x1 * cos_value));
}

static __global__ void SparkCudaMlaKvQueryRopeKernel(SparkCudaMlaKvRequest request, const uint16_t *query_pe, const uint32_t *positions, const float *cos_table, const float *sin_table, uint16_t *query_out)
{
    uint64_t pair_global_index;
    uint64_t pair_count_total;
    uint64_t row_index;
    uint64_t pair_index;
    uint64_t row_offset;
    uint64_t table_offset;
    uint32_t pair_count;
    uint32_t position;

    pair_count = request.rope_dim >> 1u;
    pair_count_total = (uint64_t)request.token_count * (uint64_t)request.query_head_count * (uint64_t)pair_count;
    pair_global_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (pair_global_index < pair_count_total)
    {
        row_index = pair_global_index / (uint64_t)pair_count;
        pair_index = pair_global_index - (row_index * (uint64_t)pair_count);
        position = positions[row_index / (uint64_t)request.query_head_count];
        if (position < request.position_count)
        {
            row_offset = row_index * (uint64_t)request.rope_dim;
            table_offset = ((uint64_t)position * (uint64_t)pair_count) + pair_index;
            SparkCudaMlaKvApplyRopePair(query_pe[row_offset + (pair_index * 2u)], query_pe[row_offset + (pair_index * 2u) + 1u], cos_table[table_offset], sin_table[table_offset], &query_out[row_offset + (pair_index * 2u)], &query_out[row_offset + (pair_index * 2u) + 1u]);
        }
        pair_global_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaMlaKvCacheWriteKernel(SparkCudaMlaKvRequest request, const uint16_t *key_pe, const uint16_t *latent, const uint32_t *positions, const uint32_t *slot_mapping, const float *cos_table, const float *sin_table, uint16_t *mla_cache)
{
    uint64_t work_index;
    uint64_t work_count;
    uint64_t work_per_token;
    uint64_t token_index;
    uint64_t inner_index;
    uint64_t cache_offset;
    uint64_t table_offset;
    uint32_t pair_count;
    uint32_t position;
    uint32_t slot_index;
    uint32_t max_slot_count;

    pair_count = request.rope_dim >> 1u;
    work_per_token = (uint64_t)request.latent_dim + (uint64_t)pair_count;
    work_count = (uint64_t)request.token_count * work_per_token;
    max_slot_count = request.block_count * request.block_size;
    work_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (work_index < work_count)
    {
        token_index = work_index / work_per_token;
        inner_index = work_index - (token_index * work_per_token);
        slot_index = slot_mapping[token_index];
        if (slot_index < max_slot_count)
        {
            cache_offset = (uint64_t)slot_index * ((uint64_t)request.latent_dim + (uint64_t)request.rope_dim);
            if (inner_index < (uint64_t)request.latent_dim)
            {
                mla_cache[cache_offset + inner_index] = latent[(token_index * (uint64_t)request.latent_dim) + inner_index];
            }
            else
            {
                inner_index -= (uint64_t)request.latent_dim;
                position = positions[token_index];
                if (position < request.position_count)
                {
                    table_offset = ((uint64_t)position * (uint64_t)pair_count) + inner_index;
                    SparkCudaMlaKvApplyRopePair(key_pe[(token_index * (uint64_t)request.rope_dim) + (inner_index * 2u)], key_pe[(token_index * (uint64_t)request.rope_dim) + (inner_index * 2u) + 1u], cos_table[table_offset], sin_table[table_offset], &mla_cache[cache_offset + (uint64_t)request.latent_dim + (inner_index * 2u)], &mla_cache[cache_offset + (uint64_t)request.latent_dim + (inner_index * 2u) + 1u]);
                }
            }
        }
        work_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaMlaRopeCacheBf16(const SparkCudaMlaKvRequest *request, const void *device_query_pe_bf16, const void *device_key_pe_bf16, const void *device_latent_bf16, const uint32_t *device_positions, const uint32_t *device_slot_mapping, const float *device_cos_table, const float *device_sin_table, void *device_query_out_bf16, void *device_mla_cache_bf16, SparkCudaMlaKvReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMlaKvRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_query_pe_bf16 == 0 || device_key_pe_bf16 == 0 || device_latent_bf16 == 0 || device_positions == 0 || device_slot_mapping == 0 || device_cos_table == 0 || device_sin_table == 0 || device_query_out_bf16 == 0 || device_mla_cache_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMlaKvFillReportShape(request, report);
    SparkCudaMlaKvQueryRopeKernel<<<SparkCudaMlaKvBlockCountHost(report->query_pair_count), SPARK_CUDA_MLA_KV_THREADS>>>(*request, (const uint16_t *)device_query_pe_bf16, device_positions, device_cos_table, device_sin_table, (uint16_t *)device_query_out_bf16);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkCudaMlaKvCacheWriteKernel<<<SparkCudaMlaKvBlockCountHost(report->cache_work_count), SPARK_CUDA_MLA_KV_THREADS>>>(*request, (const uint16_t *)device_key_pe_bf16, (const uint16_t *)device_latent_bf16, device_positions, device_slot_mapping, device_cos_table, device_sin_table, (uint16_t *)device_mla_cache_bf16);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->query_rope_kernel_count = 1u;
    report->cache_write_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
