#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_FP4_QUANT_SENTINEL 0x5350435546513431ull
#define SPARKPIPE_CUDA_FP4_QUANT_SCALE_BLOCK 16u
#define SPARKPIPE_CUDA_FP4_QUANT_FP4_MAX 6.0f
#define SPARKPIPE_CUDA_FP4_QUANT_E4M3_MAX 448.0f

typedef struct SparkCudaFp4QuantRequest
{
    uint32_t row_count;
    uint32_t col_count;
    float global_scale;
    uint64_t sentinel;
} SparkCudaFp4QuantRequest;

typedef struct SparkCudaFp4QuantReport
{
    uint64_t element_count;
    uint64_t packed_bytes;
    uint64_t scale_bytes;
    uint32_t row_count;
    uint32_t col_count;
    uint32_t scale_block_count;
    float global_scale;
    uint32_t host_reference_count;
    uint32_t quant_kernel_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
    uint32_t saturated_scale_count;
} SparkCudaFp4QuantReport;

SparkStatus SparkValidateCudaFp4QuantRequest(const SparkCudaFp4QuantRequest *request);
uint64_t SparkCudaFp4QuantPackedBytes(uint32_t rows, uint32_t cols);
uint64_t SparkCudaFp4QuantScaleBytes(uint32_t rows, uint32_t cols);
float SparkCudaFp4QuantDefaultGlobalScale(float absolute_max);
uint8_t SparkCudaFp4QuantEncodeE2m1Host(float value);
uint8_t SparkCudaFp4QuantEncodeE4m3Host(float value);
float SparkCudaFp4QuantDecodeE2m1Host(uint8_t nibble);
float SparkCudaFp4QuantDecodeE4m3Host(uint8_t byte_value);
SparkStatus SparkRunHostBf16ToFp4E2m1(const SparkCudaFp4QuantRequest *request, const uint16_t *input_bf16, uint8_t *output_fp4, uint8_t *output_scales_ue4m3, SparkCudaFp4QuantReport *report);
SparkStatus SparkRunCudaBf16ToFp4E2m1(const SparkCudaFp4QuantRequest *request, const void *device_input_bf16, void *device_output_fp4, void *device_output_scales_ue4m3, SparkCudaFp4QuantReport *report);

#ifdef __cplusplus
}
#endif
