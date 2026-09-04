#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LmGemmArguments;

int32_t Glm5NextGemmBf16(
    struct LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_bf16,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream);

uint32_t Glm5NextExpertWeightCodec(void);

int32_t Glm5NextGemmExpertWeightBf16Activation(
    struct LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_payload,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream);

#ifdef __cplusplus
}
#endif
