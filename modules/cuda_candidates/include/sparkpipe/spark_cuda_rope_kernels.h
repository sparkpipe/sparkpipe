#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_ROPE_SENTINEL 0x53504355524F5031ull

typedef struct SparkCudaRopeRequest
{
    uint32_t token_count;
    uint32_t head_count;
    uint32_t head_size;
    uint32_t rotary_dim;
    uint32_t position_count;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaRopeRequest;

typedef struct SparkCudaRopeReport
{
    uint64_t element_count;
    uint64_t rotary_pair_count;
    uint64_t copy_element_count;
    uint32_t standard_rope_kernel_count;
    uint32_t sentinel_violation_count;
} SparkCudaRopeReport;

SparkStatus SparkValidateCudaRopeRequest(const SparkCudaRopeRequest *request);
SparkStatus SparkRunCudaRopeBf16(const SparkCudaRopeRequest *request, const void *device_input_bf16, const uint32_t *device_positions, const float *device_cos_table, const float *device_sin_table, void *device_output_bf16, SparkCudaRopeReport *report);

#ifdef __cplusplus
}
#endif
