// Kimi K3. The whole model, one translation unit.
//
// TWO POOLS. Three of every four layers carry a recurrent Kimi Delta Attention
// state; the fourth attends over a KV cache. kernels/kv.cuh claimed a recurrent
// state is a pool that does not grow, and this is the first model that tests it:
// LmKvState and LmKvLatent are the same allocator, the same page table and the
// same slot accessor, differing in kGrows and a page count of one.
//
// The scheduler asks both the same questions in the same units, which is the
// property that matters - a model whose layers page differently would otherwise
// need two schedulers.
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

// K3 non-expert projections are BF16. Routed expert weights are MXFP4 while
// their activations remain BF16, so those kernels are asymmetric. A 64-element
// K tile is 128 bytes for BF16 and 32 bytes for MXFP4; both are valid TMA
// swizzle spans and share the same decoded BF16 MMA geometry.
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
// TILE_K=32 BF16-BF16 (the GEMM-008 fallback): the TP16 rank-sliced dense/routed
// projections carry inputs that are 32-multiples but not 64-multiples
// (dense_down 2112, routed_up 224, shared_w2 384), so LmGemmLaunchTileK
// re-launches these at K 32 - the smallest TMA-swizzleable BF16 row.
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 16u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 32u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmBf16Format, 64u, K3_TILE_N, 32u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 64u, K3_STAGES, K3_WARPS>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
// THE INTERLEAVED INSTANTIATIONS - pack V2 mxfp4_ws_interleaved_v1. TILE_K is
// the pack grid's 128-element k-tile (one staged box per k step, scales in
// the staged cell row), INDIRECT_A splits w1 (route-mapped latent) from w2
// (already-packed SiTU output), INTERLEAVED_B selects the cell staging.
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 32u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 64u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, true, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
template __global__ void LmGemmKernel<LmBf16Format, LmMxfp4, 16u, K3_TILE_N, 128u, K3_STAGES, K3_WARPS, false, SPARK_ACTIVATION_CODEC_NONE, true>(__grid_constant__ const LmGemmArguments, __grid_constant__ const CUtensorMap, __grid_constant__ const CUtensorMap, LmTileGeometry, LmTileGeometry, bool);
// THE TILE_K 32 INTERLEAVED INSTANTIATIONS - the TP16 pack grid (16-byte cell
// rows, SWIZZLE_NONE weight maps, single-block A rows).
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
// No rope instantiation. K3 is NoPE - the reference sets rotary_emb to None and
// carries the qk_rope slice through unrotated. This line built a rope kernel for
// a model that never rotates, which cost nothing at runtime and would have told
// whoever wrote layer.cuh that calling it was expected.
// KDA on three of every four layers, gated MLA on the fourth. The same delta
// rule Qwen 3.6 uses, at K3's widths - which is the argument that this is an
// architecture class and not one vendor's design.
// The pool stride is the outer-product slot alone; the convolution windows are
// their own pools with their own strides, and K3_KDA_STATE_BYTES is the sum a
// capacity plan budgets. If the element size ever changes, this is what stops
// the pool stride and the kernel's addressing from drifting apart again.
static_assert(K3_KDA_STATE_SLOT_BYTES >=
	(K3_KDA_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(uint16_t)),
	"a slot must hold the state it addresses");
static_assert(K3_KDA_STATE_ELEMENT_BYTES == sizeof(float),
	"the kernel addresses the pool as float; the slot must agree");
// The bf16 option is exactly half the slot and exactly that much of the state
// stream - the consumer contract the layer's fail-closed check guards.
static_assert(K3_KDA_STATE_SLOT_BYTES == 2u * K3_KDA_STATE_SLOT_BYTES_BF16,
	"the bf16 option halves the slot, no more and no less");

template __global__ void LmDeltaRuleKernel<K3_THREADS, K3_KDA_KEY_DIM, K3_KDA_VALUE_DIM>(uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const uint16_t *, const uint16_t *, const uint16_t *, const float *, const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
// The rest of the KDA path, none of which unity built: the short convolution
// with its Swish, the L2 normalisation of q and k, the bounded decay mapping,
// the output gate, and SiTU on every MLP.
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

// -- entry points ---------------------------------------------------------------
//
// Two attention kinds chosen by K3_LAYER_IS_LINEAR from the ABSOLUTE layer
// index. 93 layers with a period of four and an exception at the last one, so a
// rank that used its own offset would run the wrong kind for every layer it
// owns.

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
