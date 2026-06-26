#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_MLA_KV_SENTINEL 0x535043554D4C4131ull

typedef struct SparkCudaMlaKvRequest
{
    uint32_t token_count;
    uint32_t query_head_count;
    uint32_t latent_dim;
    uint32_t rope_dim;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t position_count;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaMlaKvRequest;

typedef struct SparkCudaMlaKvReport
{
    uint64_t query_pair_count;
    uint64_t cache_work_count;
    uint64_t cache_entry_count;
    uint32_t query_rope_kernel_count;
    uint32_t cache_write_kernel_count;
    uint32_t sentinel_violation_count;
} SparkCudaMlaKvReport;

SparkStatus SparkValidateCudaMlaKvRequest(const SparkCudaMlaKvRequest *request);
SparkStatus SparkRunCudaMlaRopeCacheBf16(const SparkCudaMlaKvRequest *request, const void *device_query_pe_bf16, const void *device_key_pe_bf16, const void *device_latent_bf16, const uint32_t *device_positions, const uint32_t *device_slot_mapping, const float *device_cos_table, const float *device_sin_table, void *device_query_out_bf16, void *device_mla_cache_bf16, SparkCudaMlaKvReport *report);

#ifdef __cplusplus
}
#endif
