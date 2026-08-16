// GLM 5.2 CUDA unity surface.
//
// The shipping precision contract is deliberately explicit:
//
//     attention, dense/shared paths, router, residuals and activations: BF16
//     routed-expert weights:                   build-selected package codec
//     routed-expert inputs and outputs:                            BF16
//     accumulators:                                                FP32
//
// Each codec is a separate AOT module and immutable package. Runtime selection
// inside a module is forbidden: startup verifies the package codec against the
// one compiled here before any weight pointer reaches a kernel.

#ifndef GLM52_EXPERT_WEIGHT_CODEC
#error "GLM52_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
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
#include "modules/glm52_resident_decode_stage/source/cuda/api.h"
#include "modules/glm52_resident_decode_stage/source/cuda/config.h"
#include "modules/glm52_resident_decode_stage/source/cuda/layer.cuh"

#define GLM52_UNITY_TILE_N 128u
#define GLM52_UNITY_TILE_K 64u
#define GLM52_UNITY_STAGES 2u
#define GLM52_UNITY_WARPS 8u

using Glm52ExpertWeightFormat =
    typename LmWeightCodec<GLM52_EXPERT_WEIGHT_CODEC>::Format;

static_assert(
    Glm52Kv::kSlotBytes == GLM52_KV_SLOT_BYTES,
    "config.h and the GLM 5.2 KV geometry disagree");
static_assert(
    GLM52_UNITY_TILE_K % LmBf16Format::kMmaK == 0u,
    "GLM 5.2 BF16 tile depth must contain complete MMA steps");
static_assert(GLM52_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_BF16,
    "GLM 5.2 routed experts require a compressed package codec");
static_assert(GLM52_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
    "GLM 5.2 routed experts require a package codec");
static_assert(
    Glm52ExpertWeightFormat::kMmaK == LmBf16Format::kMmaK,
    "GLM 5.2 expert codec must decode to the BF16 MMA geometry");
static_assert(
    LmTileKIsSwizzleable(GLM52_UNITY_TILE_K, LmBf16Format::kStoredBits),
    "GLM 5.2 BF16 activation tile must be TMA-swizzleable");

extern "C" uint32_t Glm52ExpertWeightCodec(void)
{
    return GLM52_EXPERT_WEIGHT_CODEC;
}

extern "C" int32_t Glm52GemmBf16(
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
        GLM52_UNITY_TILE_N,
        GLM52_UNITY_TILE_K,
        GLM52_UNITY_STAGES,
        GLM52_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_bf16,
            packed_rows,
            tokens,
            grouped ? GLM52_TOP_K : 1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t Glm52GemmExpertWeightBf16Activation(
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
        Glm52ExpertWeightFormat,
        GLM52_UNITY_TILE_N,
        GLM52_UNITY_STAGES,
        GLM52_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_payload,
            packed_rows,
            tokens,
            GLM52_TOP_K,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t Glm52LayerAttentionBf16(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm52LayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm52LayerDenseMlpBf16(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm52LayerDenseMlp(
        buffers,
        rows,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm52LayerMoeExpertWeightBf16Activation(
    const Glm52LayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm52LayerMoe<GLM52_EXPERT_WEIGHT_CODEC>(
        buffers,
        rows,
        packed_rows,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm52HeadFullVocab(
    const Glm52LayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return Glm52Head(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        buffers->head_vocabulary,
        rows,
        stream);
}

extern "C" int32_t Glm52HeadRestricted(
    const Glm52LayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t rows,
    cudaStream_t stream)
{
    if (token_ids == 0 || token_count == 0u ||
        token_count > GLM52_RESTRICTED_VOCAB)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return Glm52Head(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        token_ids,
        token_count,
        rows,
        stream);
}

extern "C" int32_t Glm52LayerAttentionBf16Graphed(
    LmGraphCache *graphs,
    const Glm52LayerBuffers *buffers,
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
        return Glm52LayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }

    key.rows = rows;
    // CONTRACT: the key carries no layer or buffer identity, so one cache must
    // serve exactly one layer's buffers. Sharing a cache across layers replays
    // the first-captured layer's graph - right shapes, wrong weights - and LmGraphKey
    // lives in kernels/, so the discipline lives here until a caller exists.
    key.layer_kind = 0u;
    key.format = 0u;
    key.sparse = context > GLM52_DSA_SELECTED ? 1u : 0u;
    key.context_bucket = LmGraphContextBucket(context, GLM52_DSA_SELECTED);
    if (LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK)
    {
        return LM_LAUNCH_OK;
    }
    if (LmGraphBeginCapture(stream) != LM_GRAPH_OK)
    {
        return Glm52LayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }
    status = Glm52LayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
    // Captured work does not execute: the launches above only recorded. The
    // first step of a new key must replay the graph it just built or the layer
    // silently skips attention. EndCapture failure leaves nothing to replay,
    // so fall back to running eagerly - the capture attempt itself ran nothing.
    if ( status != LM_LAUNCH_OK )
    {
        LmGraphEndCapture(graphs, &key, stream);
        return status;
    }
    if ( LmGraphEndCapture(graphs, &key, stream) != LM_GRAPH_OK )
    {
        return Glm52LayerAttention(
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
