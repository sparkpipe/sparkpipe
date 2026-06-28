#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_SPEC_VERIFY_SENTINEL 0x5350535045435631ull
#define SPARKPIPE_CUDA_SPEC_VERIFY_ERROR_COUNTERS 1u

typedef enum SparkCudaSpecVerifyErrorCounter
{
    SPARK_CUDA_SPEC_VERIFY_ERROR_SENTINEL = 0
} SparkCudaSpecVerifyErrorCounter;

typedef struct SparkCudaSpecVerifyRequest
{
    uint32_t sequence_count;
    uint32_t draft_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t sentinel;
} SparkCudaSpecVerifyRequest;

typedef struct SparkCudaSpecVerifyReport
{
    uint64_t token_count;
    uint32_t clear_kernel_count;
    uint32_t verify_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t error_counter_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaSpecVerifyReport;

SparkStatus SparkValidateCudaSpecVerifyRequest(const SparkCudaSpecVerifyRequest *request);
SparkStatus SparkRunCudaSpecVerify(const SparkCudaSpecVerifyRequest *request, const uint32_t *device_draft_token_ids, const uint32_t *device_target_token_ids, uint32_t *device_accepted_mask, uint32_t *device_accepted_prefix_lengths, uint32_t *device_error_counters, SparkCudaSpecVerifyReport *report);

#ifdef __cplusplus
}
#endif
