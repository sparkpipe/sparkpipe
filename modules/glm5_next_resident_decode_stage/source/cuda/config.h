#pragma once

// GLM 5.3 Flash (family glm5_next). Shapes and constants only.
//
// A model in this tree is a config header plus a unity build file; the
// kernels under inference/ are parameterised on the constants below.
// glm5_next is an ASSEMBLY: the MLA/indexer/MoE spine is glm52's, the KDA
// linear-attention path is k3's, hyper-connections and the kpool
// compressor mechanism are dsv4's. Every literal traces to
// model_contracts/glm53_flash_authoritative.json, held in lockstep by
// tests/test_glm5_next_geometry.py; reference semantics are transformers
// models/glm5_next/modeling_glm5_next.py.
//
// Three deltas define the family (see the name map at
// model-families/glm5_next/name_map.json):
//   1. MLA with qk_rope_head_dim = 0 - nope-only absorbed scoring, the KV
//      latent is the pure 512 lora (glm52: 576). NO rope exists anywhere
//      in the text stack (NoPE); indexer_rope_interleave is vestigial.
//   2. KDA output gate is the LOW-RANK two-stage g_a->g_b form with the
//      safe sigmoid forget gate (released K3 is full-rank g_proj).
//   3. Hybrid dispatch: 34 KDA + 11 DSA layers on a 3-dense + 42-MoE
//      skeleton, hyper-connections on every weight layer.

#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"
#include "sparkpipe/spark_glm5_next_model.h"

#define GLM5_NEXT_HIDDEN SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define GLM5_NEXT_LAYERS SPARK_GLM5_NEXT_MODEL_LAYER_COUNT
#define GLM5_NEXT_FIRST_ROUTED_LAYER SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER
#define GLM5_NEXT_VOCAB SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT
#define GLM5_NEXT_RESTRICTED_VOCAB SPARK_GLM5_NEXT_MODEL_RESTRICTED_VOCAB_COUNT
#define GLM5_NEXT_RMS_EPSILON SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON
#define GLM5_NEXT_SWIGLU_LIMIT SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT

// -- hybrid dispatch ---------------------------------------------------------
#define GLM5_NEXT_LAYER_IS_KDA(layer_index) \
	SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer_index)
#define GLM5_NEXT_LAYER_KIND(layer_index) \
	(GLM5_NEXT_LAYER_IS_KDA(layer_index) ? LM_LAYER_LINEAR : LM_LAYER_LATENT)

// -- attention: MLA with sparse selection, rope dim ZERO ----------------------
//
// The KV cache stores the shared 512-wide latent per (sequence, slot) -
// 1024 bytes at bf16, versus glm52's 1152: the rope segment is gone and
// no cache-side rope pass runs. The query is nope-only; absorbed scoring
// against the latent with scale 256**-0.5.
#define GLM5_NEXT_ATTN_HEADS SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT
#define GLM5_NEXT_LATENT SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION
#define GLM5_NEXT_ROPE_DIM SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION
#define GLM5_NEXT_QK_NOPE_DIM SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION
#define GLM5_NEXT_VALUE_DIM SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION
#define GLM5_NEXT_QUERY_A_DIM SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION
#define GLM5_NEXT_LATENT_ROW (GLM5_NEXT_LATENT + GLM5_NEXT_ROPE_DIM)
static_assert(GLM5_NEXT_ROPE_DIM == 0u,
	"glm5_next is the rope-0 instantiation; a nonzero rope dim is another family");

// Sparse selection over 4-token k-pools. The index pass scores POOLS (512
// of them at 2048 selected) mixed by the compressor, then the consumer
// expands to raw tokens; the sideband width is topk + tail.
#define GLM5_NEXT_DSA_SELECTED SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K
#define GLM5_NEXT_DSA_INDEX_HEADS SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT
#define GLM5_NEXT_DSA_INDEX_DIM SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION
#define GLM5_NEXT_DSA_QUERY_DIM SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION
#define GLM5_NEXT_DSA_INDEX_EPSILON SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON
#define GLM5_NEXT_DSA_INDEX_SCALE SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE
#define GLM5_NEXT_DSA_INDEX_HEAD_WEIGHT_SCALE \
	SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE
#define GLM5_NEXT_DSA_KPOOL SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL

// -- KDA linear attention (k3 donor) ------------------------------------------
//
// Kimi delta rule: qk L2-normalised in kernel, beta per head through
// sigmoid, per-head-PER-CHANNEL bounded decay, fp32 state. The pack ships
// the fused q|k|v|beta tensor and the fused decay|gate-down bottleneck;
// the gate is completed by the up-projection (g_b) below.
#define GLM5_NEXT_KDA_HEADS SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT
#define GLM5_NEXT_KDA_KEY_DIM SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION
#define GLM5_NEXT_KDA_VALUE_DIM SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION
#define GLM5_NEXT_KDA_QK_DIM SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION
#define GLM5_NEXT_KDA_CONV_KERNEL SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL
#define GLM5_NEXT_KDA_GATE_LOWER_BOUND SPARK_GLM5_NEXT_MODEL_KDA_GATE_LOWER_BOUND
#define GLM5_NEXT_KDA_LOW_RANK 128u
#define GLM5_NEXT_KDA_STATE_ELEMENT_BYTES SPARK_GLM5_NEXT_MODEL_KDA_STATE_ELEMENT_BYTES
#define GLM5_NEXT_KDA_STATE_BYTES_PER_LAYER SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER

// -- hyper-connections (dsv4 donor) -------------------------------------------
//
// Four residual streams; the mix projection reads the flattened
// unweighted-RMSNorm'd streams; sinkhorn 20 on the comb matrix. The final
// head collapse is an UNWEIGHTED MEAN then RMSNorm.
#define GLM5_NEXT_HC SPARK_GLM5_NEXT_MODEL_HC_MULT
#define GLM5_NEXT_HC_MIX SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION
#define GLM5_NEXT_HC_FN_COLUMNS SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS
#define GLM5_NEXT_HC_SINKHORN_ITERATIONS SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS
#define GLM5_NEXT_HC_EPSILON SPARK_GLM5_NEXT_MODEL_HC_EPSILON
#define GLM5_NEXT_HC_FLAT (GLM5_NEXT_HC * GLM5_NEXT_HIDDEN)

// -- MoE ----------------------------------------------------------------------
//
// 288 routed + 1 shared at top-8; sigmoid router with the frozen
// e_score_correction bias (noaux_tc, n_group 1), fp32 logits, norm top-k,
// routed scaling 2.5. Dense MLP (int 12288) on the first three layers.
#define GLM5_NEXT_EXPERTS SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define GLM5_NEXT_TOP_K SPARK_GLM5_NEXT_MODEL_MOE_TOP_K
#define GLM5_NEXT_EXPERT_INTERMEDIATE SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION
#define GLM5_NEXT_DENSE_INTERMEDIATE SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define GLM5_NEXT_ROUTED_SCALE SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR
#define GLM5_NEXT_W1_COMPONENTS SPARK_GLM5_NEXT_MODEL_MOE_W1_COMPONENT_COUNT

// -- quantisation ---------------------------------------------------------------
#define GLM5_NEXT_FP8_SCALE_BLOCK SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK

// -- speculative decode ----------------------------------------------------------
#define GLM5_NEXT_MTP_LAYER_INDEX SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX

// -- KV geometry -------------------------------------------------------------
//
// DSA slots: the pure latent. KDA carries no KV rows; its state is the
// fp32 slab above plus the three conv windows (q|k|v concatenated,
// kernel 4 wide, bf16). The indexer cache row packs [k | gate | valid].
#define GLM5_NEXT_KV_BITS 16u
#define GLM5_NEXT_KV_PAGE_SLOTS 64u
#define GLM5_NEXT_KV_SLOT_BYTES SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES
#define GLM5_NEXT_KDA_CONV_WINDOW_BYTES_PER_LAYER \
	(GLM5_NEXT_KDA_QK_DIM * 3u * GLM5_NEXT_KDA_CONV_KERNEL * 2u)
#define GLM5_NEXT_INDEX_ROW_BYTES \
	(SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u)

// -- derived -----------------------------------------------------------------
#define GLM5_NEXT_ROUTED_LAYERS (GLM5_NEXT_LAYERS - GLM5_NEXT_FIRST_ROUTED_LAYER)
#define GLM5_NEXT_GATE_UP_DIM (GLM5_NEXT_EXPERT_INTERMEDIATE * GLM5_NEXT_W1_COMPONENTS)
#define GLM5_NEXT_WEIGHT_LAYERS SPARK_GLM5_NEXT_MODEL_WEIGHT_LAYER_COUNT

static inline uint32_t Glm5NextRowsPerExpert(uint32_t tokens)
{
	return(((tokens * GLM5_NEXT_TOP_K) + GLM5_NEXT_EXPERTS - 1u) / GLM5_NEXT_EXPERTS);
}
