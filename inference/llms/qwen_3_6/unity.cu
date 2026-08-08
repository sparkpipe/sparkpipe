// Qwen 3.6. The whole model, one translation unit.
//
// Gated DeltaNet on 48 of 64 layers, full attention on 16. The same two-pool
// shape K3 has, reached from a different vendor - which is the argument that
// this is an architecture class and not one company's choice, and the reason
// kernels/kv.cuh parameterises growth rather than special-casing it.
//
// The GDN state carries a short causal convolution window alongside the
// delta-rule matrix. Both are per-sequence, neither grows, so both live in one
// non-growing slot and the pool does not need to know which is which.
#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/llms/qwen_3_6/layer.cuh"


static_assert(Qwen36GdnState::kGrows == false, "GDN state is fixed per sequence");
static_assert(QWEN36_NOPE_DIM + QWEN36_ROPE_DIM == QWEN36_HEAD_DIM,
	"the decode kernel splits a head into nope and rope; they must be the head");
static_assert(QWEN36_LAYER_IS_LINEAR(0) && !QWEN36_LAYER_IS_LINEAR(3),
	"period 4, full attention in phase 3");

#define QWEN36_TILE_N 128u
#define QWEN36_STAGES 2u
#define QWEN36_WARPS 8u
#define QWEN36_THREADS 256u

template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, QWEN36_TILE_N, 64u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, QWEN36_TILE_N, 64u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, QWEN36_TILE_N, 64u, QWEN36_STAGES, QWEN36_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmFusedResidualRmsNormKernel<QWEN36_THREADS,uint16_t>(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, float);
template __global__ void LmSiluMulKernel<QWEN36_THREADS>(const uint16_t *, uint16_t *, uint32_t, bool);
template __global__ void LmRopePerHeadKernel<QWEN36_THREADS>(uint16_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, float);
template __global__ void LmSplitQkvKernel<QWEN36_THREADS>(const uint16_t *, LmQkvLayout, uint16_t *, uint16_t *, uint16_t *, uint32_t, float);
// The gated attention path: the query|gate de-interleave and the output gate.
template __global__ void LmSplitQueryGateKernel<QWEN36_THREADS>(const uint16_t *, uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmOutputGateKernel<QWEN36_THREADS>(uint16_t *, const uint16_t *, uint32_t);
// The linear layers. 48 of 64, with a fixed state instead of a growing cache.
//
// The state and the convolution window share one non-growing slot, which is why
// QWEN36_GDN_STATE_BYTES is their sum and kernels/kv.cuh sizes the pool from it.
template __global__ void LmDeltaRuleKernel<QWEN36_THREADS, QWEN36_GDN_KEY_DIM, QWEN36_GDN_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
// The GDN gate producer: beta and decay logits to retention factors and
// write strengths, per value head.
template __global__ void LmGdnGateKernel<QWEN36_THREADS, QWEN36_GDN_KEY_DIM>(const uint16_t *, const uint16_t *, const float *, const float *, float *, float *, uint32_t, uint32_t);
template __global__ void LmCausalConvKernel<QWEN36_THREADS, QWEN36_GDN_CONV_KERNEL, LM_CONV_SWISH,uint16_t>(uint16_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmExpandHeadsKernel<QWEN36_THREADS>(const uint16_t *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
// Per-head KV: the pack store and the GQA decode, not the MLA latent pair -
// the latent kernel cannot express a value that is not a prefix of the key.
template __global__ void LmGqaKvStoreKernel<Qwen36FullKv, QWEN36_THREADS, QWEN36_KV_HEADS, QWEN36_HEAD_DIM, QWEN36_HEAD_DIM>(LmKvView, const uint16_t *, const uint16_t *, const uint32_t *, const uint32_t *, uint32_t);
template __global__ void LmGqaAttentionDecodeKernel<Qwen36FullKv, QWEN36_THREADS, QWEN36_KV_HEADS, QWEN36_HEAD_DIM, QWEN36_HEAD_DIM>(const uint16_t *, LmKvView, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, float, uint16_t *, const uint32_t *);
template __global__ void LmHeadCandidateKernel<QWEN36_THREADS, 1024u>(const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadCommitKernel<QWEN36_THREADS>(const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
template __global__ void LmMoeFinalizeKernel<QWEN36_THREADS>(const uint16_t *, const uint32_t *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t);

extern "C" int32_t Qwen36GemmBf16(
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
        QWEN36_TILE_N,
        LmBf16Format::kTileK,
        QWEN36_STAGES,
        QWEN36_WARPS>(
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

// -- entry points ---------------------------------------------------------------
//
// Two layer kinds, chosen by the host from the layer index through
// QWEN36_LAYER_IS_LINEAR. Separate entry points rather than a flag: the state
// pool and the KV pool are different geometries, and that belongs in the type.

extern "C" int32_t Qwen36LayerLinearBf16(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerLinear<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen36LayerAttentionBf16(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerAttention<LmBf16Format,Qwen36FullKv>(b,rows,context,sms,s));
}

extern "C" int32_t Qwen36LayerDenseMlpBf16(const Qwen36LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t s)
{
	return(Qwen36LayerDenseMlp<LmBf16Format>(b,rows,sms,s));
}

extern "C" int32_t Qwen36HeadFullVocab(const Qwen36LayerBuffers *b, const void *norm_weight, const void *head_weight, uint32_t rows, cudaStream_t s)
{
	return(Qwen36Head(b,norm_weight,head_weight,0,QWEN36_VOCAB,rows,s));
}

extern "C" int32_t Qwen36HeadRestricted(const Qwen36LayerBuffers *b, const void *norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t count, uint32_t rows, cudaStream_t s)
{
	return(Qwen36Head(b,norm_weight,head_weight,token_ids,count,rows,s));
}

