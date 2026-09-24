
#ifndef LAGUNA_EXPERT_WEIGHT_CODEC
#error "LAGUNA_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
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
#include "modules/laguna_resident_decode_stage/source/cuda/api.h"
#include "modules/laguna_resident_decode_stage/source/cuda/config.h"
#include "modules/laguna_resident_decode_stage/source/cuda/layer.cuh"
#define SPARK_FAMILY_CAMEL Laguna
#define SPARK_FAMILY_UPPER LAGUNA
#define SPARK_FAMILY_LOWER laguna

#include "sparkpipe/family/spark_family.h"

#define LAGUNA_UNITY_TILE_N 128u
#define LAGUNA_UNITY_TILE_K 64u
#define LAGUNA_UNITY_STAGES 2u
#define LAGUNA_UNITY_WARPS 8u

using LagunaExpertWeightFormat =
    typename LmWeightCodec<LAGUNA_EXPERT_WEIGHT_CODEC>::Format;

static_assert(
    LagunaKv::kSlotBytes == LAGUNA_KV_SLOT_BYTES,
    "config.h and the laguna KV geometry disagree");
static_assert(
    LAGUNA_UNITY_TILE_K % LmBf16Format::kMmaK == 0u,
    "laguna BF16 tile depth must contain complete MMA steps");
static_assert(LAGUNA_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
    "laguna routed experts require a package codec");
static_assert(
    LagunaExpertWeightFormat::kMmaK == LmBf16Format::kMmaK,
    "laguna expert codec must decode to the BF16 MMA geometry");
static_assert(
    LmTileKIsSwizzleable(LAGUNA_UNITY_TILE_K, LmBf16Format::kStoredBits),
    "laguna BF16 activation tile must be TMA-swizzleable");

extern "C" int32_t LagunaLayerAttentionBf16(
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    return LagunaLayerAttention(
        buffers,
        rows,
        context,
        multiprocessors,
        stream);
}

extern "C" int32_t LagunaHeadFullVocab(
    const LagunaLayerBuffers *buffers,
    const void *norm_weight_bf16,
    const void *head_weight_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    return LagunaHead(
        buffers,
        norm_weight_bf16,
        head_weight_bf16,
        0,
        LAGUNA_VOCAB,
        rows,
        stream);
}

extern "C" int32_t LagunaLayerAttentionBf16Graphed(
    LmGraphCache *graphs,
    const LagunaLayerBuffers *buffers,
    uint32_t rows,
    uint32_t context,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGraphKey key;
    int32_t status;

    if (graphs == 0)
    {
        return LagunaLayerAttention(
            buffers,
            rows,
            context,
            multiprocessors,
            stream);
    }

    key.rows = rows;
    key.layer_kind = LAGUNA_LAYER_IS_SLIDING(buffers->layer_index) ? 1u : 0u;
    key.format = 0u;
    key.sparse = LAGUNA_LAYER_IS_SLIDING(buffers->layer_index) ? 1u : 0u;
    key.context_bucket = LmGraphContextBucket(context, LAGUNA_WINDOW);
    if (LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK)
    {
        return LM_LAUNCH_OK;
    }
    if (LmGraphBeginCapture(stream) != LM_GRAPH_OK)
    {
        return LagunaLayerAttention(
            buffers,
            rows,
            context,
            multiprocessors,
            stream);
    }
    status = LagunaLayerAttention(
        buffers,
        rows,
        context,
        multiprocessors,
        stream);
    if ( status != LM_LAUNCH_OK )
    {
        LmGraphEndCapture(graphs, &key, stream);
        return status;
    }
    if ( LmGraphEndCapture(graphs, &key, stream) != LM_GRAPH_OK )
    {
        return LagunaLayerAttention(
            buffers,
            rows,
            context,
            multiprocessors,
            stream);
    }
    return LmGraphReplay(graphs, &key, stream) == LM_GRAPH_OK
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

#include "sparkpipe/family/glm/spark_glm_unity_gemm.cuh"

#include "sparkpipe/family/glm/spark_glm_unity_head_restricted.cuh"
