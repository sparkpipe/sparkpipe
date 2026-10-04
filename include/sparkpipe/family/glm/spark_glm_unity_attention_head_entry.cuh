#pragma once

extern "C" int32_t SPARK_FAMILY_BARE(LayerAttentionBf16)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return SPARK_FAMILY_BARE(LayerAttention)(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
}

extern "C" int32_t SPARK_FAMILY_BARE(HeadFullVocab)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return SPARK_FAMILY_BARE(Head)(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        buffers->head_vocabulary,
        rows,
        stream);
}
