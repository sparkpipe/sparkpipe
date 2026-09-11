
#ifndef LING_EXPERT_WEIGHT_CODEC
#error "LING_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif

#include "runtime/gemm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/graph.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/speculate.cuh"
#include "inference/kernels/topk.cuh"
#include "modules/ling_resident_decode_stage/source/cuda/api.h"
#include "modules/ling_resident_decode_stage/source/cuda/config.h"
#include "modules/ling_resident_decode_stage/source/cuda/layer.cuh"

#define LING_UNITY_TILE_N 128u
#define LING_UNITY_TILE_K 64u
#define LING_UNITY_STAGES 2u
#define LING_UNITY_WARPS 8u

using LingExpertWeightFormat =
    typename LmWeightCodec<LING_EXPERT_WEIGHT_CODEC>::Format;

static_assert(
    LingKv::kSlotBytes == LING_KV_SLOT_BYTES,
    "config.h and the ling KV geometry disagree");
static_assert(
    LING_UNITY_TILE_K % LmBf16Format::kMmaK == 0u,
    "ling BF16 tile depth must contain complete MMA steps");
static_assert(LING_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
    "ling routed experts require a package codec");
static_assert(
    LingExpertWeightFormat::kMmaK == LmBf16Format::kMmaK,
    "ling expert codec must decode to the BF16 MMA geometry");
static_assert(
    LmTileKIsSwizzleable(LING_UNITY_TILE_K, LmBf16Format::kStoredBits),
    "ling BF16 activation tile must be TMA-swizzleable");

extern "C" uint32_t LingExpertWeightCodec(void)
{
    return LING_EXPERT_WEIGHT_CODEC;
}

extern "C" int32_t LingGemmBf16(
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
        LING_UNITY_TILE_N,
        LING_UNITY_TILE_K,
        LING_UNITY_STAGES,
        LING_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_bf16,
            packed_rows,
            tokens,
            grouped ? LING_TOP_K : 1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t LingGemmExpertWeightBf16Activation(
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
        LingExpertWeightFormat,
        LING_UNITY_TILE_N,
        LING_UNITY_STAGES,
        LING_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_payload,
            packed_rows,
            tokens,
            LING_TOP_K,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t LingLayerAttentionBf16(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return LingLayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
}

extern "C" int32_t LingLayerDenseMlpBf16(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return LingLayerDenseMlp(
        buffers,
        rows,
        multiprocessors,
        stream);
}

extern "C" int32_t LingLayerMoeExpertWeightBf16Activation(
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return LingLayerMoe<LING_EXPERT_WEIGHT_CODEC>(
        buffers,
        rows,
        packed_rows,
        multiprocessors,
        stream);
}

extern "C" int32_t LingHeadFullVocab(
    const LingLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return LingHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        buffers->head_vocabulary,
        rows,
        stream);
}

extern "C" int32_t LingLayerAttentionBf16Graphed(
    LmGraphCache *graphs,
    const LingLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGraphKey key;
    int32_t status;

    if (graphs == 0)
    {
        return LingLayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }

    key.rows = rows;
    key.layer_kind = 0u;
    key.format = 0u;
    key.sparse = 0u;
    key.context_bucket = LmGraphContextBucket(context, 0u);
    if (LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK)
    {
        return LM_LAUNCH_OK;
    }
    if (LmGraphBeginCapture(stream) != LM_GRAPH_OK)
    {
        return LingLayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }
    status = LingLayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
    if ( status != LM_LAUNCH_OK )
    {
        LmGraphEndCapture(graphs, &key, stream);
        return status;
    }
    if ( LmGraphEndCapture(graphs, &key, stream) != LM_GRAPH_OK )
    {
        return LingLayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }
    return LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}
