#pragma once

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void SPARK_FAMILY_BARE(SplitFusedProjectionsKernel)(
    const uint16_t *__restrict__ qkvb_bf16,
    uint16_t *__restrict__ query_bf16,
    uint16_t *__restrict__ key_bf16,
    uint16_t *__restrict__ value_bf16,
    uint16_t *__restrict__ beta_bf16,
    uint32_t rows,
    uint32_t qk_dim,
    uint32_t v_dim,
    uint32_t heads,
    uint32_t fused_rows)
{
    uint32_t row = blockIdx.x, index;
    const uint32_t k_offset = qk_dim;
    const uint32_t v_offset = 2u * qk_dim;
    const uint32_t beta_offset = v_offset + v_dim;
    uint64_t fused = (uint64_t)row * fused_rows;
    uint64_t dense = (uint64_t)row * qk_dim;
    if (row >= rows)
        return;
    for (index = threadIdx.x; index < qk_dim; index += THREADS)
        query_bf16[dense + index] = qkvb_bf16[fused + index];
    for (index = threadIdx.x; index < qk_dim; index += THREADS)
        key_bf16[dense + index] = qkvb_bf16[fused + k_offset + index];
    for (index = threadIdx.x; index < v_dim; index += THREADS)
        value_bf16[((uint64_t)row * v_dim) + index] =
            qkvb_bf16[fused + v_offset + index];
    for (index = threadIdx.x; index < heads; index += THREADS)
        beta_bf16[((uint64_t)row * heads) + index] =
            qkvb_bf16[fused + beta_offset + index];
}

static int32_t SPARK_FAMILY_BARE(DeltaRuleOptIn)(uint32_t shared_bytes)
{
    return(LmKernelSharedMemoryOptIn(
        (const void *)LmDeltaRuleKernel<SPARK_FAMILY_BARE_CONST(LAYER_THREADS),
                                        SPARK_FAMILY_BARE_CONST(KDA_KEY_DIM),
                                        SPARK_FAMILY_BARE_CONST(KDA_VALUE_DIM)>,
        shared_bytes));
}
