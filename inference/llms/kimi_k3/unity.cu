#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/kimi_k3/layer.cuh"

using K3LinearState = LmKvState<K3_KDA_STATE_SLOT_BYTES>;

static_assert(K3LinearState::kGrows == false, "a delta-rule state does not grow with context");
static_assert(K3LinearState::kPageSlots == 1u, "one slot per sequence");
static_assert(K3GlobalKv::kGrows == true, "the global layers still page");

#define K3_TILE_N 128u
#define K3_STAGES 2u
#define K3_WARPS 8u
#define K3_THREADS 256u

template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void K3EmbeddingKernel<K3_THREADS>(const uint16_t *, const uint32_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmFusedResidualRmsNormKernel<K3_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmFusedResidualRmsNormKernel<K3_THREADS,float>(const uint16_t *, const uint16_t *, const float *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<K3_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
static_assert(K3_KDA_STATE_SLOT_BYTES >=
	(K3_KDA_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(uint16_t)),
	"a slot must hold the state it addresses");
static_assert(K3_KDA_STATE_ELEMENT_BYTES == sizeof(float),
	"the kernel addresses the pool as float; the slot must agree");
static_assert(K3_KDA_STATE_SLOT_BYTES == 2u * K3_KDA_STATE_SLOT_BYTES_BF16,
	"the bf16 option halves the slot, no more and no less");

template __global__ void LmDeltaRuleKernel<K3_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void LmCausalConvKernel<K3_THREADS, K3_KDA_CONV_KERNEL, LM_CONV_SWISH,float>(uint16_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmL2NormalisePerHeadKernel<K3_THREADS, K3_KDA_KEY_DIM>(uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmBoundedDecayKernel<K3_THREADS, K3_KDA_KEY_DIM>(const uint16_t *, const float *, const float *, float *, uint32_t, float, uint32_t);
template __global__ void LmOutputGateKernel<K3_THREADS>(uint16_t *, const uint16_t *, uint32_t);
template __global__ void LmSituMulKernel<K3_THREADS>(const uint16_t *, uint16_t *, uint32_t, float, float);
template __global__ void LmKvStoreKernel<K3GlobalKv, K3_THREADS>(LmKvView, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t);
template __global__ void LmHeadCandidateKernel<K3_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<K3_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmMoeFinalizeKernel<K3_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);

template __global__ void LmAttentionDecodeKernel<K3GlobalKv, K3_THREADS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM>(const uint16_t *, const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmTopkSmallKernel<K3_THREADS, K3_TOP_K, true, 1u, 1u, LM_TOPK_SCORE_SIGMOID>(const float *, uint32_t, uint32_t *, float *, const float *, const uint16_t *, float);
template __global__ void LmSigmoidRowsKernel<K3_THREADS>(const uint16_t *, float *, uint32_t);


extern "C" int32_t K3HeadFullVocab(const K3LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t s)
{
	return(K3Head(b,norm_weight,head_weight,0,K3_VOCAB,rows,s));
}

extern "C" int32_t K3HeadRestricted(const K3LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t s)
{
	return(K3Head(b,norm_weight,head_weight,token_ids,count,rows,s));
}

template __global__ void LmPerHeadProjectKernel<K3_THREADS, K3_KV_LORA_RANK, K3_V_HEAD_DIM>(const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t);

template __global__ void LmAttnResKernel<K3_THREADS, K3_ATTNRES_MAX_SOURCES>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, float);

template __global__ void LmCopyRowsKernel<K3_THREADS>(const uint16_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmGatherRowsKernel<K3_THREADS>(const uint16_t *, const uint32_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmRouteBuildKernel<K3_THREADS, K3_EXPERTS>(const uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t, uint32_t *);
template __global__ void LmGemmTilePrefixKernel<32u>(const uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t *);
