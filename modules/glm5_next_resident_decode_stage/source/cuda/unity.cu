
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
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next

#include "sparkpipe/family/spark_family.h"

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

#include "sparkpipe/family/glm/spark_glm_unity_gemm.cuh"

#include "sparkpipe/family/glm/spark_glm_unity_glm5_next_ling.cuh"

#include "sparkpipe/family/glm/spark_glm_unity_head_restricted.cuh"
