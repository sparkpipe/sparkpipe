#pragma once

extern "C" int32_t SPARK_FAMILY_BARE(HeadRestricted)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t rows,
    cudaStream_t stream)
{
    if (token_ids == 0 || token_count == 0u ||
        token_count > SPARK_FAMILY_BARE_CONST(RESTRICTED_VOCAB))
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return SPARK_FAMILY_BARE(Head)(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        token_ids,
        token_count,
        rows,
        stream);
}
