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

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
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
#include "modules/glm5_next_resident_decode_stage/source/cuda/api.h"
#include "modules/glm5_next_resident_decode_stage/source/cuda/config.h"
#include "modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh"

#define GLM5_NEXT_UNITY_TILE_N 128u
#define GLM5_NEXT_UNITY_TILE_K 64u
#define GLM5_NEXT_UNITY_STAGES 2u
#define GLM5_NEXT_UNITY_WARPS 8u

using Glm5NextExpertWeightFormat =
    typename LmWeightCodec<GLM5_NEXT_EXPERT_WEIGHT_CODEC>::Format;

static_assert(
    Glm5NextKv::kSlotBytes == GLM5_NEXT_KV_SLOT_BYTES,
    "config.h and the GLM 5.2 KV geometry disagree");
static_assert(
    GLM5_NEXT_UNITY_TILE_K % LmBf16Format::kMmaK == 0u,
    "GLM 5.2 BF16 tile depth must contain complete MMA steps");
static_assert(GLM5_NEXT_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_BF16,
    "GLM 5.2 routed experts require a compressed package codec");
static_assert(GLM5_NEXT_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
    "GLM 5.2 routed experts require a package codec");
static_assert(
    Glm5NextExpertWeightFormat::kMmaK == LmBf16Format::kMmaK,
    "GLM 5.2 expert codec must decode to the BF16 MMA geometry");
static_assert(
    LmTileKIsSwizzleable(GLM5_NEXT_UNITY_TILE_K, LmBf16Format::kStoredBits),
    "GLM 5.2 BF16 activation tile must be TMA-swizzleable");

extern "C" uint32_t Glm5NextExpertWeightCodec(void)
{
    return GLM5_NEXT_EXPERT_WEIGHT_CODEC;
}

extern "C" int32_t Glm5NextGemmBf16(
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
        GLM5_NEXT_UNITY_TILE_N,
        GLM5_NEXT_UNITY_TILE_K,
        GLM5_NEXT_UNITY_STAGES,
        GLM5_NEXT_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_bf16,
            packed_rows,
            tokens,
            grouped ? GLM5_NEXT_TOP_K : 1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t Glm5NextGemmExpertWeightBf16Activation(
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
        Glm5NextExpertWeightFormat,
        GLM5_NEXT_UNITY_TILE_N,
        GLM5_NEXT_UNITY_STAGES,
        GLM5_NEXT_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_payload,
            packed_rows,
            tokens,
            GLM5_NEXT_TOP_K,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t Glm5NextLayerAttentionBf16(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm5NextLayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm5NextLayerDenseMlpBf16(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm5NextLayerDenseMlp(
        buffers,
        rows,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm5NextLayerMoeExpertWeightBf16Activation(
    const Glm5NextLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return Glm5NextLayerMoe<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(
        buffers,
        rows,
        packed_rows,
        multiprocessors,
        stream);
}

extern "C" int32_t Glm5NextHeadFullVocab(
    const Glm5NextLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return Glm5NextHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        buffers->head_vocabulary,
        rows,
        stream);
}

extern "C" int32_t Glm5NextHeadCertifiedB1(
    const Glm5NextLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    const uint8_t *certified_payload,
    const float *certified_scale_f32,
    const float *certified_norm_f32,
    void *certified_scratch,
    uint32_t *candidate_ids,
    uint32_t *screened_count,
    uint32_t rank_offset,
    uint32_t vocabulary,
    cudaStream_t stream)
{
    return Glm5NextHeadCertifiedB1(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        certified_payload,
        certified_scale_f32,
        certified_norm_f32,
        certified_scratch,
        candidate_ids,
        screened_count,
        rank_offset,
        vocabulary,
        stream);
}

extern "C" cudaError_t SparkGlm5NextLaunchHeadCertifiedQuantize(
    cudaStream_t stream,
    const void *head_bf16,
    uint8_t *certified_payload,
    float *certified_scale_f32,
    float *certified_norm_f32,
    uint32_t vocabulary,
    uint32_t hidden_dimension)
{
    return SparkLmHostLaunchHeadCertifiedFp8Quantize(
        stream, head_bf16, certified_payload, certified_scale_f32,
        certified_norm_f32, vocabulary, hidden_dimension);
}

extern "C" int32_t Glm5NextHeadRestricted(
    const Glm5NextLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t rows,
    cudaStream_t stream)
{
    if (token_ids == 0 || token_count == 0u ||
        token_count > GLM5_NEXT_RESTRICTED_VOCAB)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return Glm5NextHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        token_ids,
        token_count,
        rows,
        stream);
}

extern "C" int32_t Glm5NextLayerAttentionBf16Graphed(
    LmGraphCache *graphs,
    const Glm5NextLayerBuffers *buffers,
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
        return Glm5NextLayerAttention(
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
    key.sparse = context > GLM5_NEXT_DSA_SELECTED ? 1u : 0u;
    key.context_bucket = LmGraphContextBucket(context, GLM5_NEXT_DSA_SELECTED);
    if (LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK)
    {
        return LM_LAUNCH_OK;
    }
    if (LmGraphBeginCapture(stream) != LM_GRAPH_OK)
    {
        return Glm5NextLayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }
    status = Glm5NextLayerAttention(
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
        return Glm5NextLayerAttention(
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
