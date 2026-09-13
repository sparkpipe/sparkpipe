#pragma once

// MiMo 2.5. Shapes and constants only.
//
// AUDIT INCOMPLETE, AND MY FIRST PASS AT IT WAS WRONG. I recorded here that this
// matched XiaomiMiMo/MiMo-V2.5-Pro at 70 layers and 384 routed experts. The
// constants below say 48 layers, hidden 4096, 256 experts. Those are different
// models, and the test_layer_kinds gate printed the layer count next to the
// kinds, which is how the contradiction surfaced.
//
// So: the shapes here match SOME MiMo checkpoint and it is not Pro. The 6:1
// window ratio, window 128 and top-8 routing are shared across the V2.5 family,
// which is why nothing looked wrong. Whoever knows which checkpoint this was
// taken from should name it, the way qwen_3_6 names the 27B.
//
// ONE GAP THE SHAPES DO NOT SHOW: the reference keeps long-context quality at a
// ~7x smaller KV cache "via learnable attention sink bias". That is a per-head
// bias admitted into the softmax denominator, and LmAttentionDecodeKernel has
// no sink term. Without it the 6:1 window ratio is a cache saving with nothing
// compensating for what the window drops - the shape is right and the numbers
// will not be. See docs/MODEL_SUPPORT.md.
//
// The second model in this tree, and the test of whether kernels/ is actually
// model-agnostic or merely glm52 with the names filed off. Nothing in kernels/
// changed to accommodate it.
//
// What differs from GLM 5.2 is real: no latent compression, so the cache holds
// per-head key and value rows; two attention branches with different rope thetas
// and different KV head counts; a sliding window on one of them. All of it is
// expressible as a geometry and two arguments.

#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

#define MIMO25_HIDDEN 4096u                    /* CONFIG hidden_size */
#define MIMO25_LAYERS 48u                      /* CONFIG num_hidden_layers */
#define MIMO25_VOCAB 152576u                   /* CONFIG vocab_size */
#define MIMO25_RMS_EPSILON 1e-5f               /* CONFIG layernorm_epsilon */

// -- attention: two branches over stored KV, no latent -------------------------
//
// The full branch attends over the whole context; the SWA branch only over the
// last SLIDING_WINDOW tokens. They differ in KV head count and rope theta, which
// is why the geometry is instantiated twice rather than parameterised at
// runtime - the slot size differs, and that sizes the pool.

#define MIMO25_ATTN_HEADS 64u                  /* CONFIG num_attention_heads */
#define MIMO25_HEAD_DIM 192u                   /* CONFIG head_dim */
#define MIMO25_VALUE_DIM 128u                  /* CONFIG v_head_dim */
#define MIMO25_ROPE_DIM 64u                    /* head_dim * partial_rotary_factor */
#define MIMO25_FULL_KV_HEADS 4u
#define MIMO25_SWA_KV_HEADS 8u
#define MIMO25_SLIDING_WINDOW 128u
#define MIMO25_FULL_ROPE_THETA 10000000.0f
#define MIMO25_SWA_ROPE_THETA 10000.0f
#define MIMO25_VALUE_SCALE 0.707f

// The QKV projection is FUSED - one GEMM producing query, key and value
// concatenated - where GLM 5.2 uses four separate projections. The widths
// confirm it: 12288 query plus 4 KV heads of (192 key + 128 value) is 13568 on a
// full layer, and 8 KV heads gives 14848 on a sliding-window one.
//
// Layers are one kind or the other, not both, so the projection width and the
// rope theta are per-layer rather than per-model.
#define MIMO25_Q_DIM 12288u                    /* CONFIG heads * head_dim */
#define MIMO25_FULL_QKV_DIM 13568u             /* Q + 4 * (192 + 128) */
#define MIMO25_SWA_QKV_DIM 14848u              /* Q + 8 * (192 + 128) */
#define MIMO25_O_INPUT_DIM 8192u               /* heads * v_head_dim */
// INFERRED, NOT READ FROM A CONFIG. Every model in this tree with both a dense
// and a routed MLP puts the dense layers first, and MiMo's family card says one
// dense layer then the rest MoE. This constant is that pattern, and it is a
// GUESS in the sense kimi_k3's header uses the word: a number inferred from a
// lineage is not the same kind of fact as one read from a published config, and
// losing that distinction is how an inferred number becomes load-bearing.
//
// It is load-bearing now - bind.cu branches on it - so it needs confirming
// against whichever checkpoint these 48 layers came from. SETTLED BY: that
// checkpoint's config.json, field first_k_dense_replace or its equivalent. One
// grep, once someone names the variant - and naming the variant is the blocker,
// not the grep, because the shapes here match no MiMo card I have seen.
#define MIMO25_FIRST_ROUTED_LAYER 1u           /* GUESS (family pattern) */


// -- MoE -----------------------------------------------------------------------
//
// Same 256 experts at top-8 as GLM 5.2, so rows per expert is batch/32 and the
// tile selector produces the same heights. Two models sharing a routing shape
// share every GEMM instantiation.

#define MIMO25_EXPERTS 256u                    /* CONFIG n_routed_experts */
#define MIMO25_TOP_K 8u                        /* CONFIG num_experts_per_tok */
#define MIMO25_EXPERT_INTERMEDIATE 2048u
#define MIMO25_DENSE_INTERMEDIATE 16384u

// -- KV geometry ---------------------------------------------------------------
//
// Per-head key and value rows, not a latent. That is the cost of no absorption:
// 4 full KV heads at 192+128 dims is 2,560 bytes per slot against GLM 5.2's
// 1,152, on a model with a quarter the hidden size.
//
// The SWA branch stores more heads but bounds its pool by the window, so its
// resident cost is 8 * 320 * 2 * 128 tokens rather than growing with context.

#define MIMO25_KV_BITS 16u
#define MIMO25_KV_PAGE_SLOTS 64u

// -- derived -------------------------------------------------------------------

#define MIMO25_GATE_UP_DIM (MIMO25_EXPERT_INTERMEDIATE * 2u)
#define MIMO25_ROPE_HALF (MIMO25_ROPE_DIM / 2u)

static inline uint32_t Mimo25RowsPerExpert(uint32_t tokens)
{
	return(((tokens * MIMO25_TOP_K) + MIMO25_EXPERTS - 1u) / MIMO25_EXPERTS);
}

// -- layer kinds --
// Sliding window and global attention interleaved at 6:1, so a period of
// seven with the global layer last. MIMO25_LAYER_KIND_FULL and _SWA above are
// the old names for these and are the enum this replaces.
#define MIMO25_ATTENTION_PERIOD 7u
#define MIMO25_LAYER_KIND(layer) \
	((((layer) % MIMO25_ATTENTION_PERIOD) == (MIMO25_ATTENTION_PERIOD - 1u)) \
		? LM_LAYER_FULL : LM_LAYER_WINDOW)
