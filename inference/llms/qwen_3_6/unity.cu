#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/llms/qwen_3_6/layer.cuh"


static_assert(Qwen38_27bGdnState::kGrows == false, "GDN state is fixed per sequence");
static_assert(QWEN38_27B_NOPE_DIM + QWEN38_27B_ROPE_DIM == QWEN38_27B_HEAD_DIM,
	"the decode kernel splits a head into nope and rope; they must be the head");
static_assert(QWEN38_27B_LAYER_IS_LINEAR(0) && !QWEN38_27B_LAYER_IS_LINEAR(3),
	"period 4, full attention in phase 3");

#define QWEN38_27B_TILE_N 128u
#define QWEN38_27B_STAGES 2u
#define QWEN38_27B_WARPS 8u
#define QWEN38_27B_THREADS 256u

template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, QWEN38_27B_TILE_N, 64u, QWEN38_27B_STAGES, QWEN38_27B_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, QWEN38_27B_TILE_N, 64u, QWEN38_27B_STAGES, QWEN38_27B_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, QWEN38_27B_TILE_N, 64u, QWEN38_27B_STAGES, QWEN38_27B_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmFusedResidualRmsNormKernel<QWEN38_27B_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<QWEN38_27B_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmRopePerHeadKernel<QWEN38_27B_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmSplitQkvKernel<QWEN38_27B_THREADS>(const uint16_t *, LmQkvLayout, uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmSplitQueryGateKernel<QWEN38_27B_THREADS>(const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmOutputGateKernel<QWEN38_27B_THREADS>(uint16_t *, const uint16_t *, uint32_t);
template __global__ void LmDeltaRuleKernel<QWEN38_27B_THREADS, QWEN38_27B_GDN_KEY_DIM, QWEN38_27B_GDN_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void LmGdnGateKernel<QWEN38_27B_THREADS, QWEN38_27B_GDN_KEY_DIM>(const uint16_t *, const uint16_t *, const float *, const float *, float *, float *, uint32_t, uint32_t);
template __global__ void LmCausalConvKernel<QWEN38_27B_THREADS, QWEN38_27B_GDN_CONV_KERNEL, LM_CONV_SWISH,uint16_t>(uint16_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmExpandHeadsKernel<QWEN38_27B_THREADS>(const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void LmGqaKvStoreKernel<Qwen38_27bFullKv, QWEN38_27B_THREADS, QWEN38_27B_KV_HEADS, QWEN38_27B_HEAD_DIM, QWEN38_27B_HEAD_DIM>(LmKvView, const uint16_t *, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t);
template __global__ void LmGqaAttentionDecodeKernel<Qwen38_27bFullKv, QWEN38_27B_THREADS, QWEN38_27B_KV_HEADS, QWEN38_27B_HEAD_DIM, QWEN38_27B_HEAD_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmHeadCandidateKernel<QWEN38_27B_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<QWEN38_27B_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmMoeFinalizeKernel<QWEN38_27B_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);

extern "C" int32_t Qwen38_27bGemmBf16(
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
        QWEN38_27B_TILE_N,
        LmBf16Format::kTileK,
        QWEN38_27B_STAGES,
        QWEN38_27B_WARPS>(
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


extern "C" int32_t Qwen38_27bLayerLinearBf16(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen38_27bLayerLinear<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen38_27bLayerAttentionBf16(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t s)
{
	return(Qwen38_27bLayerAttention<LmBf16Format,Qwen38_27bFullKv>(b,rows,context,sms,s));
}

extern "C" int32_t Qwen38_27bLayerDenseMlpBf16(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen38_27bLayerDenseMlp<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen38_27bHeadFullVocab(const Qwen38_27bLayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t s)
{
	return(Qwen38_27bHead(b,norm_weight,head_weight,0,QWEN38_27B_VOCAB,rows,s));
}

extern "C" int32_t Qwen38_27bHeadRestricted(const Qwen38_27bLayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t s)
{
	return(Qwen38_27bHead(b,norm_weight,head_weight,token_ids,count,rows,s));
}

