#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_ACTIVATION_SENTINEL 0x535043554143544Full

typedef enum SparkCudaGatedActivationKind
{
    SPARK_CUDA_GATED_ACTIVATION_SILU = 1,
    SPARK_CUDA_GATED_ACTIVATION_GELU = 2,
    SPARK_CUDA_GATED_ACTIVATION_GELU_TANH = 3
} SparkCudaGatedActivationKind;

typedef struct SparkCudaActivationRequest
{
    uint32_t row_count;
    uint32_t hidden_size;
    uint32_t input_stride;
    uint32_t output_stride;
    SparkCudaGatedActivationKind activation_kind;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaActivationRequest;

typedef struct SparkCudaActivationReport
{
    uint64_t element_count;
    uint32_t row_count;
    uint32_t hidden_size;
    uint32_t gated_activation_kernel_count;
    uint32_t activation_kind;
    uint32_t sentinel_violation_count;
} SparkCudaActivationReport;

SparkStatus SparkValidateCudaActivationRequest(const SparkCudaActivationRequest *request);
SparkStatus SparkRunCudaGatedActivationBf16(const SparkCudaActivationRequest *request, const void *device_gate_up_bf16, void *device_output_bf16, SparkCudaActivationReport *report);

#ifdef __cplusplus
}
#endif
