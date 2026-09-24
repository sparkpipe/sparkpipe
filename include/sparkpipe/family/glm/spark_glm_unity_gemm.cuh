#pragma once

extern "C" uint32_t SPARK_FAMILY_BARE(ExpertWeightCodec)(void)
{
    return SPARK_FAMILY_BARE_CONST(EXPERT_WEIGHT_CODEC);
}

extern "C" int32_t SPARK_FAMILY_BARE(GemmBf16)(
    LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_bf16,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream_handle)
{
    return LmGemmLaunch<
        LmBf16Format,
        SPARK_FAMILY_BARE_CONST(UNITY_TILE_N),
        SPARK_FAMILY_BARE_CONST(UNITY_TILE_K),
        SPARK_FAMILY_BARE_CONST(UNITY_STAGES),
        SPARK_FAMILY_BARE_CONST(UNITY_WARPS)>(
            arguments,
            activation_bf16,
            weight_bf16,
            packed_rows,
            tokens,
            grouped ? SPARK_FAMILY_BARE_CONST(TOP_K) : 1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t SPARK_FAMILY_BARE(GemmExpertWeightBf16Activation)(
    LmGemmArguments *arguments,
    const void *activation_bf16,
    const void *weight_payload,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    void *stream_handle)
{
    if (!grouped)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return LmGemmWeightOnlyLaunch<
        SPARK_FAMILY_BARE(ExpertWeightFormat),
        SPARK_FAMILY_BARE_CONST(UNITY_TILE_N),
        SPARK_FAMILY_BARE_CONST(UNITY_STAGES),
        SPARK_FAMILY_BARE_CONST(UNITY_WARPS)>(
            arguments,
            activation_bf16,
            weight_payload,
            packed_rows,
            tokens,
            SPARK_FAMILY_BARE_CONST(TOP_K),
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t SPARK_FAMILY_BARE(LayerDenseMlpBf16)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return SPARK_FAMILY_BARE(LayerDenseMlp)(
        buffers,
        rows,
        multiprocessors,
        stream);
}

extern "C" int32_t SPARK_FAMILY_BARE(LayerMoeExpertWeightBf16Activation)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return SPARK_FAMILY_BARE(LayerMoe)<SPARK_FAMILY_BARE_CONST(EXPERT_WEIGHT_CODEC)>(
        buffers,
        rows,
        packed_rows,
        multiprocessors,
        stream);
}
