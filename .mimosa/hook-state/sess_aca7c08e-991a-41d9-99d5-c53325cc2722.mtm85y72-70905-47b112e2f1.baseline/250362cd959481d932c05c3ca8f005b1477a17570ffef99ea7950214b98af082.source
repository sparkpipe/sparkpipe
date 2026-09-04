#pragma once
#if defined(__CUDACC__)
#include "sparkpipe/spark_lm_kernels.cuh"
#else
#include <stdint.h>
#include <stddef.h>
#ifndef LM_HOST_CUDA_PRELUDE
typedef int cudaError_t;
typedef struct { unsigned x, y, z; } dim3;
#define cudaSuccess 0
#endif
static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8Quantize(
    dim3 stream, const void *head_bf16, uint8_t *shadow_payload,
    float *shadow_scale_f32, float *cert_norm_f32,
    uint32_t candidate_count, uint32_t hidden_dimension)
{ (void)stream; (void)head_bf16; (void)shadow_payload;
  (void)shadow_scale_f32; (void)cert_norm_f32; (void)candidate_count;
  (void)hidden_dimension; return cudaSuccess; }
static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
    dim3 stream, const void *hidden_bf16, const void *head_weight_bf16,
    const uint8_t *shadow_payload, const float *shadow_scale_f32,
    const float *cert_norm_f32, void *scratch, uint32_t *candidate_ids,
    uint32_t *candidate_count, uint32_t *output_token_id,
    float *output_score, uint32_t candidate_offset, uint32_t row_count,
    uint32_t vocabulary_count, uint32_t hidden_dimension)
{ (void)stream; (void)hidden_bf16; (void)head_weight_bf16;
  (void)shadow_payload; (void)shadow_scale_f32; (void)cert_norm_f32;
  (void)scratch; (void)candidate_ids; (void)candidate_count;
  (void)output_token_id; (void)output_score; (void)candidate_offset;
  (void)row_count; (void)vocabulary_count; (void)hidden_dimension;
  return cudaSuccess; }
#endif
