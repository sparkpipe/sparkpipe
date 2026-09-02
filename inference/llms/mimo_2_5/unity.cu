
#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/mimo_2_5/config.h"
#include "inference/llms/mimo_2_5/layer.cuh"


static_assert(Mimo25FullKv::kSlotBytes == 2560u, "4 heads x (192 key + 128 value) x bf16");
static_assert(Mimo25SwaKv::kSlotBytes == 5120u, "8 heads, twice the slot");

#define MIMO25_TILE_N 128u
#define MIMO25_STAGES 2u
#define MIMO25_WARPS 8u
#define MIMO25_THREADS 256u


template __global__ void LmGemmKernel<LmFp8, LmFp8, 16u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmFp8, LmFp8, 32u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmFp8, LmFp8, 64u, MIMO25_TILE_N, 128u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmInt7, LmInt7, 16u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmInt7, LmInt7, 32u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmInt7, LmInt7, 64u, MIMO25_TILE_N, 256u, MIMO25_STAGES, MIMO25_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);

template __global__ void LmFusedResidualRmsNormKernel<MIMO25_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<MIMO25_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmQuantiseRowsKernel<LmFp8, MIMO25_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);
template __global__ void LmQuantiseRowsKernel<LmInt7, MIMO25_THREADS>(const uint16_t *, const uint32_t *, uint8_t *, uint8_t *, uint32_t, uint32_t);

template __global__ void LmRopeKernel<MIMO25_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);

template __global__ void LmGqaAttentionDecodeKernel<Mimo25FullKv, MIMO25_THREADS, MIMO25_FULL_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_VALUE_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmGqaAttentionDecodeKernel<Mimo25SwaKv, MIMO25_THREADS, MIMO25_SWA_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_VALUE_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);

template __global__ void LmTopkSmallKernel<MIMO25_THREADS, MIMO25_TOP_K, false, 1u, 1u, LM_TOPK_SCORE_IDENTITY>(const float *, uint32_t, uint32_t *, float *, const float *, const uint16_t *, float);
template __global__ void LmRouteBuildKernel<MIMO25_THREADS, MIMO25_EXPERTS>(const uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t, uint32_t *, uint32_t, uint32_t *);
template __global__ void LmSplitQkvKernel<MIMO25_THREADS>(const uint16_t *, LmQkvLayout, uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
template __global__ void LmRopePerHeadKernel<MIMO25_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmGqaKvStoreKernel<Mimo25FullKv, MIMO25_THREADS, MIMO25_FULL_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_VALUE_DIM>(LmKvView, const uint16_t *, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t);
template __global__ void LmGqaKvStoreKernel<Mimo25SwaKv, MIMO25_THREADS, MIMO25_SWA_KV_HEADS, MIMO25_HEAD_DIM, MIMO25_VALUE_DIM>(LmKvView, const uint16_t *, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t);
template __global__ void LmHeadCandidateKernel<MIMO25_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<MIMO25_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);


extern "C" int32_t Mimo25GemmFp8(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmFp8,MIMO25_TILE_N,128u,MIMO25_STAGES,MIMO25_WARPS>(
		args,a,b,packed_rows,tokens,MIMO25_TOP_K,groups,k,n,sms,grouped,stream));
}

extern "C" int32_t Mimo25GemmInt7(LmGemmArguments *args, const void *a, const void *b,
	uint32_t packed_rows, uint32_t tokens, uint32_t groups,
	uint32_t k, uint32_t n, uint32_t sms, bool grouped, cudaStream_t stream)
{
	return(LmGemmLaunch<LmInt7,MIMO25_TILE_N,256u,MIMO25_STAGES,MIMO25_WARPS>(
		args,a,b,packed_rows,tokens,MIMO25_TOP_K,groups,k,n,sms,grouped,stream));
}


extern "C" int32_t Mimo25LayerAttentionFullFp8(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerAttention<LmFp8,Mimo25FullKv,MIMO25_FULL_KV_HEADS,MIMO25_FULL_QKV_DIM>(
		b,rows,context,0u,MIMO25_FULL_ROPE_THETA,sms,stream));
}

extern "C" int32_t Mimo25LayerAttentionSwaFp8(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerAttention<LmFp8,Mimo25SwaKv,MIMO25_SWA_KV_HEADS,MIMO25_SWA_QKV_DIM>(
		b,rows,context,MIMO25_SLIDING_WINDOW,MIMO25_SWA_ROPE_THETA,sms,stream));
}

extern "C" int32_t Mimo25LayerAttentionFullInt7(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerAttention<LmInt7,Mimo25FullKv,MIMO25_FULL_KV_HEADS,MIMO25_FULL_QKV_DIM>(
		b,rows,context,0u,MIMO25_FULL_ROPE_THETA,sms,stream));
}

extern "C" int32_t Mimo25LayerAttentionSwaInt7(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerAttention<LmInt7,Mimo25SwaKv,MIMO25_SWA_KV_HEADS,MIMO25_SWA_QKV_DIM>(
		b,rows,context,MIMO25_SLIDING_WINDOW,MIMO25_SWA_ROPE_THETA,sms,stream));
}

extern "C" int32_t Mimo25LayerMoeFp8(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerMoe<LmFp8>(b,rows,packed_rows,sms,stream));
}

extern "C" int32_t Mimo25LayerMoeInt7(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerMoe<LmInt7>(b,rows,packed_rows,sms,stream));
}

extern "C" int32_t Mimo25LayerDenseMlpFp8(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	return(Mimo25LayerDenseMlp<LmFp8>(b,rows,sms,stream));
}

extern "C" int32_t Mimo25HeadFullVocab(const Mimo25LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t stream)
{
	return(Mimo25Head(b,norm_weight,head_weight,0,MIMO25_VOCAB,rows,stream));
}

extern "C" int32_t Mimo25HeadRestricted(const Mimo25LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t stream)
{
	return(Mimo25Head(b,norm_weight,head_weight,token_ids,count,rows,stream));
}

