#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/route.cuh"
#include "inference/llms/qwen_3_8/layer.cuh"


static_assert(Qwen38GdnState::kGrows == false, "GDN state is fixed per sequence");
static_assert(QWEN38_NOPE_DIM + QWEN38_ROPE_DIM == QWEN38_HEAD_DIM,
	"the decode kernel splits a head into nope and rope; they must be the head");
static_assert(QWEN38_LAYER_IS_LINEAR(0) && !QWEN38_LAYER_IS_LINEAR(3),
	"period 4, full attention in phase 3");

#define QWEN38_TILE_N 128u
#define QWEN38_STAGES 2u
#define QWEN38_WARPS 8u
#define QWEN38_THREADS 256u

template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmFp8, 16u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmFp8, 32u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmFp8, 64u, QWEN38_TILE_N, 64u, QWEN38_STAGES, QWEN38_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmFusedResidualRmsNormKernel<QWEN38_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<QWEN38_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmRopePerHeadKernel<QWEN38_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmSplitQkvKernel<QWEN38_THREADS>(const uint16_t *, LmQkvLayout, uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSplitQueryGateKernel<QWEN38_THREADS>(const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmOutputGateKernel<QWEN38_THREADS>(uint16_t *, const uint16_t *, uint32_t);
template __global__ void Qwen38HeadRmsNormKernel<QWEN38_THREADS>(const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmDeltaRuleKernel<QWEN38_THREADS, QWEN38_GDN_KEY_DIM, QWEN38_GDN_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void LmGdnGateKernel<QWEN38_THREADS, QWEN38_GDN_KEY_DIM>(const uint16_t *, const uint16_t *, const float *, const float *, float *, float *, uint32_t, uint32_t);
template __global__ void LmCausalConvKernel<QWEN38_THREADS, QWEN38_GDN_CONV_KERNEL, LM_CONV_SWISH,uint16_t>(uint16_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmExpandHeadsKernel<QWEN38_THREADS>(const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void Qwen38GatedHeadNormKernel<QWEN38_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmGqaKvStoreKernel<Qwen38FullKv, QWEN38_THREADS, QWEN38_KV_HEADS, QWEN38_HEAD_DIM, QWEN38_HEAD_DIM>(LmKvView, const uint16_t *, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t);
template __global__ void LmGqaAttentionDecodeKernel<Qwen38FullKv, QWEN38_THREADS, QWEN38_KV_HEADS, QWEN38_HEAD_DIM, QWEN38_HEAD_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmTopkSmallKernel<QWEN38_THREADS, QWEN38_TOP_K, true, 1u, 1u, LM_TOPK_SCORE_IDENTITY>(const float *, uint32_t, uint32_t *, float *, const float *, const uint16_t *, float);
template __global__ void LmRouteBuildKernel<QWEN38_THREADS, QWEN38_EXPERTS>(const uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t, uint32_t *);
template __global__ void LmMoeFinalizeKernel<QWEN38_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmCopyRowsKernel<QWEN38_THREADS>(const uint16_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void Qwen38SharedExpertAddKernel<QWEN38_THREADS>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t);
template __global__ void LmHeadCandidateKernel<QWEN38_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<QWEN38_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);

extern "C" int32_t Qwen38GemmBf16(
    LmGemmArguments *arguments,
    const void *activation,
    const void *weight,
    uint32_t row_count,
    uint32_t token_count,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessor_count,
    bool grouped,
    cudaStream_t stream)
{
    return LmGemmLaunch<
        LmBf16Format,
        QWEN38_TILE_N,
        LmBf16Format::kTileK,
        QWEN38_STAGES,
        QWEN38_WARPS>(
            arguments,
            activation,
            weight,
            row_count,
            token_count,
            1u,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessor_count,
            grouped,
            stream);
}


extern "C" int32_t Qwen38LayerLinearBf16(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen38LayerLinear<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen38LayerAttentionBf16(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t s)
{
	return(Qwen38LayerAttention<LmBf16Format,Qwen38FullKv>(b,rows,context,sms,s));
}

extern "C" int32_t Qwen38LayerMoeBf16(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen38LayerMoe<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen38HeadFullVocab(const Qwen38LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t s)
{
	return(Qwen38Head(b,norm_weight,head_weight,0,QWEN38_VOCAB,rows,s));
}

extern "C" int32_t Qwen38HeadRestricted(const Qwen38LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t s)
{
	return(Qwen38Head(b,norm_weight,head_weight,token_ids,count,rows,s));
}
