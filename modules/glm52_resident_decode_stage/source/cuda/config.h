#pragma once

// GLM 5.2. Shapes and constants only.
//
// A model in this tree is a config header plus a unity build file. Everything
// else comes from kernels/, parameterised on the constants below. Adding GLM 5.3
// means adding a directory with two files, not a tree - the old layout had
// glm52 spread across five directories and 109 files, none of which named
// itself a driver.
//
// Anything in this directory beyond config.h and unity.cu is a claim that
// something about this model genuinely cannot be expressed as a parameter. That
// claim should be rare and should say why in the file that makes it.
//
// Values marked CONFIG come from the published model config. Nothing here is
// derived from a measurement, and nothing here is a guess - a guessed constant
// belongs in the file of the model it was guessed for, marked, not here.

#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"
#include "sparkpipe/spark_glm52_model.h"

#define GLM52_HIDDEN SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define GLM52_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define GLM52_FIRST_ROUTED_LAYER SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER
#define GLM52_VOCAB SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT
// The constrained-decoding subset. A caller with a grammar or a tool schema pays
// this instead of the vocabulary: 256 tokens is 1.6 MB of embedding against
// 1.9 GB, and it is exact rather than approximate.
#define GLM52_RESTRICTED_VOCAB SPARK_GLM52_MODEL_RESTRICTED_VOCAB_COUNT
#define GLM52_RMS_EPSILON SPARK_GLM52_MODEL_RMS_NORM_EPSILON
#define GLM52_ROPE_THETA SPARK_GLM52_MODEL_ROPE_THETA

// -- attention: MLA with sparse selection ------------------------------------
//
// The KV cache stores a shared latent row per (sequence, slot) rather than
// per-head key and value rows. That is the whole reason this model is bandwidth
// tractable at decode: the cache read per slot is 1152 bytes instead of 57 KB,
// at the cost of small per-head GEMMs to reconstruct - a large byte reduction
// for negligible added compute, which is the right trade on a bandwidth-bound
// machine.

#define GLM52_ATTN_HEADS SPARK_GLM52_MODEL_HEAD_COUNT
#define GLM52_LATENT SPARK_GLM52_MODEL_LATENT_DIMENSION
#define GLM52_ROPE_DIM SPARK_GLM52_MODEL_ROPE_DIMENSION
#define GLM52_QK_NOPE_DIM SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION
#define GLM52_VALUE_DIM SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION
#define GLM52_QUERY_A_DIM SPARK_GLM52_MODEL_QUERY_A_DIMENSION

// Sparse selection. The index pass scores all slots with a cheap low-rank head
// and keeps the top GLM52_DSA_SELECTED; the full attention then reads only
// those. Index state is shared across a group of layers, so the score is
// computed once per group rather than once per layer.
#define GLM52_DSA_SELECTED SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define GLM52_DSA_INDEX_HEADS SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT
#define GLM52_DSA_INDEX_DIM SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION
#define GLM52_DSA_QUERY_DIM (GLM52_DSA_INDEX_HEADS * GLM52_DSA_INDEX_DIM)
#define GLM52_DSA_INDEX_EPSILON SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON
#define GLM52_DSA_INDEX_SCALE SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE

// -- MoE ---------------------------------------------------------------------
//
// 256 experts at top-8 means rows per expert at decode is batch/32. That number
// drives the GEMM tile height and is the reason TILE_M is selected per token
// bucket rather than fixed: at B1024 it is 32 rows, and a 16-row tile would
// split every expert and double the weight stream.

#define GLM52_EXPERTS SPARK_GLM52_MODEL_MOE_EXPERT_COUNT
#define GLM52_TOP_K SPARK_GLM52_MODEL_MOE_TOP_K
#define GLM52_EXPERT_INTERMEDIATE SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION
#define GLM52_DENSE_INTERMEDIATE SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define GLM52_ROUTED_SCALE SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR
#define GLM52_W1_COMPONENTS SPARK_GLM52_MODEL_MOE_W1_COMPONENT_COUNT

// -- quantisation ------------------------------------------------------------
//
// Group sizes are format properties, not model choices, but they are stated
// here because the GEMM's K tile must be a whole swizzle span in BYTES and that
// depends on the element width. NVFP4 at 4 bits needs a 256-element K tile
// where FP8 needs 128; kernels/tile.cuh asserts it.

#define GLM52_FP8_SCALE_BLOCK SPARK_GLM52_MODEL_FP8_SCALE_BLOCK
#define GLM52_NVFP4_GROUP SPARK_GLM52_MODEL_NVFP4_GROUP_SIZE
#define GLM52_MXFP4_GROUP SPARK_GLM52_MODEL_MXFP4_GROUP_SIZE

// -- speculative decode ------------------------------------------------------

#define GLM52_MTP_DRAFT_TOKENS SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT
#define GLM52_MTP_LAYER_INDEX SPARK_GLM52_MODEL_MTP_LAYER_INDEX

// -- KV geometry -------------------------------------------------------------
//
// The cache stores opaque bytes; this is the only place the model's slot layout
// is stated. kernels/kv.cuh turns it into a type, and every allocator, page
// table and eviction path is shared with every other model regardless of what a
// slot means to it.
//
// A page is the unit of prefix sharing between sequences, so it is sized to be
// large enough that the page table is not itself a significant read and small
// enough that a short sequence does not waste one.

// The cache format is chosen independently of the weight format. A slot is read
// once per (sequence, position) and shared with nothing, where a weight tile is
// read once and shared across every row in it - so the two sit at different
// points on the precision-versus-bytes curve. BF16 here is the conservative
// choice; FP8 halves it and is a one-line change.
#define GLM52_KV_BITS 16u
#define GLM52_KV_PAGE_SLOTS 64u
#define GLM52_KV_SLOT_BYTES ((GLM52_LATENT_ROW * GLM52_KV_BITS) / 8u)

// -- derived -----------------------------------------------------------------
//
// Written once here rather than recomputed at call sites. A dimension derived
// twice is a dimension that can disagree with itself.

#define GLM52_ROUTED_LAYERS (GLM52_LAYERS - GLM52_FIRST_ROUTED_LAYER)
#define GLM52_GATE_UP_DIM (GLM52_EXPERT_INTERMEDIATE * GLM52_W1_COMPONENTS)
#define GLM52_LATENT_ROW (GLM52_LATENT + GLM52_ROPE_DIM)
#define GLM52_WEIGHT_LAYERS SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT

// Rows one expert receives on average at a given batch. The GEMM tile height is
// selected from this, so it is defined where the constants are rather than in
// the launcher.
static inline uint32_t Glm52RowsPerExpert(uint32_t tokens)
{
	return(((tokens * GLM52_TOP_K) + GLM52_EXPERTS - 1u) / GLM52_EXPERTS);
}

static inline uint32_t Glm52LayerHasFullIndexer(uint32_t layer)
{
	return(layer < 3u || (layer >= 6u && ((layer - 6u) % SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT) == 0u) ? 1u : 0u);
}

// -- layer kinds --
// Uniform: every layer is latent-absorbed MLA with the DSA index. The only one
// of the six that does not alternate, which is why it never needed to say so.
#define GLM52_LAYER_KIND(layer) ((void)(layer), LM_LAYER_LATENT)
