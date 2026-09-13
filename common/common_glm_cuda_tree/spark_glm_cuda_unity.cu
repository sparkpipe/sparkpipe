
#ifndef GLM_EXPERT_WEIGHT_CODEC
#error "GLM_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
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
#include "common/common_glm_cuda_tree/spark_glm_cuda_api.h"
#include "common/common_glm_cuda_tree/spark_glm_cuda_config.h"
#include "common/common_glm_cuda_tree/spark_glm_cuda_layer.cuh"

#define GLM_UNITY_TILE_N SPARK_LLM_TILE_N
#define GLM_UNITY_TILE_K SPARK_LLM_TILE_K
#define GLM_UNITY_STAGES SPARK_LLM_TILE_STAGES
#define GLM_UNITY_WARPS SPARK_LLM_TILE_WARPS

using GlmExpertWeightFormat =
    typename LmWeightCodec<GLM_EXPERT_WEIGHT_CODEC>::Format;

static_assert(
    GlmKv::kSlotBytes == GLM_KV_SLOT_BYTES,
    "spark_glm_cuda_config.h and the GLM KV geometry disagree");
static_assert(
    GLM_UNITY_TILE_K % LmBf16Format::kMmaK == 0u,
    "GLM BF16 tile depth must contain complete MMA steps");
static_assert(GLM_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
    "GLM routed experts require a package codec");
static_assert(
    GlmExpertWeightFormat::kMmaK == LmBf16Format::kMmaK,
    "GLM expert codec must decode to the BF16 MMA geometry");
static_assert(
    LmTileKIsSwizzleable(GLM_UNITY_TILE_K, LmBf16Format::kStoredBits),
    "GLM BF16 activation tile must be TMA-swizzleable");

extern "C" uint32_t GlmExpertWeightCodec(void)
{
    return GLM_EXPERT_WEIGHT_CODEC;
}

extern "C" int32_t GlmGemmBf16(
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
        GLM_UNITY_TILE_N,
        GLM_UNITY_TILE_K,
        GLM_UNITY_STAGES,
        GLM_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_bf16,
            packed_rows,
            tokens,
            grouped ? GLM_TOP_K : 1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t GlmGemmExpertWeightBf16Activation(
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
        GlmExpertWeightFormat,
        GLM_UNITY_TILE_N,
        GLM_UNITY_STAGES,
        GLM_UNITY_WARPS>(
            arguments,
            activation_bf16,
            weight_payload,
            packed_rows,
            tokens,
            GLM_TOP_K,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            (cudaStream_t)stream_handle);
}

extern "C" int32_t GlmLayerAttentionBf16(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t layer_in_group,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return GlmLayerAttention(
        buffers,
        rows,
        context,
        layer_in_group,
        multiprocessors,
        stream);
}

extern "C" int32_t GlmLayerDenseMlpBf16(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return GlmLayerDenseMlp(
        buffers,
        rows,
        multiprocessors,
        stream);
}

extern "C" int32_t GlmLayerMoeExpertWeightBf16Activation(
    const GlmLayerBuffers *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return GlmLayerMoe<GLM_EXPERT_WEIGHT_CODEC>(
        buffers,
        rows,
        packed_rows,
        multiprocessors,
        stream);
}

extern "C" int32_t GlmHeadFullVocab(
    const GlmLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return GlmHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        buffers->head_vocabulary,
        rows,
        stream);
}

extern "C" cudaError_t SparkGlmLaunchHeadCertifiedQuantize(cudaStream_t stream,const void *head_bf16,uint8_t *certified_payload,float *certified_scale_f32,float *certified_norm_f32,uint32_t vocabulary,uint32_t hidden_dimension)
{
    return SparkLmHostLaunchHeadCertifiedFp8Quantize(stream,head_bf16,certified_payload,certified_scale_f32,certified_norm_f32,vocabulary,hidden_dimension);
}

extern "C" int32_t GlmHeadRestricted(
    const GlmLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t rows,
    cudaStream_t stream)
{
    if (token_ids == 0 || token_count == 0u ||
        token_count > GLM_RESTRICTED_VOCAB)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return GlmHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        token_ids,
        token_count,
        rows,
        stream);
}

extern "C" int32_t GlmLayerAttentionBf16Graphed(
    LmGraphCache *graphs,
    const GlmLayerBuffers *buffers,
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
        return GlmLayerAttention(
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
    key.sparse = context > GLM_DSA_SELECTED ? 1u : 0u;
    key.context_bucket = LmGraphContextBucket(context, GLM_DSA_SELECTED);
    if (LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK)
    {
        return LM_LAUNCH_OK;
    }
    if (LmGraphBeginCapture(stream) != LM_GRAPH_OK)
    {
        return GlmLayerAttention(
            buffers,
            rows,
            context,
            layer_in_group,
            multiprocessors,
            stream);
    }
    status = GlmLayerAttention(
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
        return GlmLayerAttention(
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
