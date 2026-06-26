#include <string.h>

#include "sparkpipe/spark_cuda_mla_kv_kernels.h"

static uint64_t SparkCudaMlaKvQueryPairCount(const SparkCudaMlaKvRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * (uint64_t)request->query_head_count * ((uint64_t)request->rope_dim / 2u);
}

static uint64_t SparkCudaMlaKvCacheWorkCount(const SparkCudaMlaKvRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * ((uint64_t)request->latent_dim + ((uint64_t)request->rope_dim / 2u));
}

static uint64_t SparkCudaMlaKvCacheEntryCount(const SparkCudaMlaKvRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->latent_dim + (uint64_t)request->rope_dim;
}

SparkStatus SparkValidateCudaMlaKvRequest(const SparkCudaMlaKvRequest *request)
{
    uint64_t query_pair_count;
    uint64_t cache_work_count;
    uint64_t cache_entry_count;
    uint64_t cache_capacity_count;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->query_head_count == 0u || request->latent_dim == 0u || request->rope_dim == 0u || request->block_size == 0u || request->block_count == 0u || request->position_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_MLA_KV_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->rope_dim & 1u) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    query_pair_count = SparkCudaMlaKvQueryPairCount(request);
    cache_work_count = SparkCudaMlaKvCacheWorkCount(request);
    cache_entry_count = SparkCudaMlaKvCacheEntryCount(request);
    cache_capacity_count = (uint64_t)request->block_count * (uint64_t)request->block_size * cache_entry_count;
    if (query_pair_count == 0u || cache_work_count == 0u || cache_entry_count == 0u || cache_capacity_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkFillCudaMlaKvReportShape(const SparkCudaMlaKvRequest *request, SparkCudaMlaKvReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->query_pair_count = SparkCudaMlaKvQueryPairCount(request);
    report->cache_work_count = SparkCudaMlaKvCacheWorkCount(request);
    report->cache_entry_count = SparkCudaMlaKvCacheEntryCount(request);
}

SparkStatus SparkRunCudaMlaRopeCacheBf16(const SparkCudaMlaKvRequest *request, const void *device_query_pe_bf16, const void *device_key_pe_bf16, const void *device_latent_bf16, const uint32_t *device_positions, const uint32_t *device_slot_mapping, const float *device_cos_table, const float *device_sin_table, void *device_query_out_bf16, void *device_mla_cache_bf16, SparkCudaMlaKvReport *report)
{
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
    SparkFillCudaMlaKvReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
