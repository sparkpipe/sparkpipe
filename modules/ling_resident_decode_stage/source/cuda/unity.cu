
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
#include "inference/kernels/topk.cuh"
#include "modules/ling_resident_decode_stage/source/cuda/api.h"
#include "modules/ling_resident_decode_stage/source/cuda/config.h"
#include "modules/ling_resident_decode_stage/source/cuda/layer.cuh"
#define SPARK_FAMILY_CAMEL Ling
#define SPARK_FAMILY_UPPER LING
#define SPARK_FAMILY_LOWER ling

#include "sparkpipe/family/spark_family.h"

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

#include "sparkpipe/family/glm/spark_glm_unity_gemm.cuh"

#include "sparkpipe/family/glm/spark_glm_unity_attention_head_entry.cuh"
