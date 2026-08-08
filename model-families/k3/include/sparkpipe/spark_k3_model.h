// Kimi K3's model geometry for the host tiers, mirroring the audited values
// in inference/llms/kimi_k3/config.h. The firmware config is the source of
// truth; tests/test_model_families.py holds every literal below against it
// in lockstep.
//
// This header replaces the pre-audit constant set the recovered resident
// decode stage module was written against. Every corrected value was a
// K2-lineage guess: 72 layers -> 93, KDA heads 64 -> 96, shared experts
// 1 -> 2, expert intermediate 2048 -> 3072, dense intermediate
// 18432 -> 33792, routed scale 2.5 -> 1.0, MLA rope-dim 0 with a
// 1/sqrt(128) scale -> an UNROTATED 64-wide slice with 1/sqrt(192), and
// uniform 9-layer AttnRes blocks -> blocks of 12 with a partial final
// block. The module still speaks this header's vocabulary, so the names
// stay and the values move; where the audit changed the SEMANTICS (Stable
// LatentMoE projections, KDA convolution windows, the full-rank gate, the
// transposed fp32 state) the vocabulary grew instead - each addition names
// its config.h source in a comment.
#ifndef SPARKPIPE_SPARK_K3_MODEL_H
#define SPARKPIPE_SPARK_K3_MODEL_H

#include <stdint.h>

// One geometry source for the byte-level KDA slab and the MLA cache: this
// shim aliases spark_k3_kv_geometry.h's layout instead of re-deriving it,
// and the compile-time checks at the bottom hold the two headers together.
#include "sparkpipe/spark_k3_kv_geometry.h"

/* K3_HIDDEN / K3_LAYERS / K3_VOCAB / K3_MAX_CONTEXT / K3_FIRST_ROUTED_LAYER. */
#define SPARK_K3_MODEL_HIDDEN_DIMENSION 7168u              /* K3_HIDDEN */
#define SPARK_K3_MODEL_LAYER_COUNT 93u                     /* K3_LAYERS (was a 72-layer K2-lineage guess) */
#define SPARK_K3_MODEL_FIRST_ROUTED_LAYER 1u               /* K3_FIRST_ROUTED_LAYER */
#define SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS 1048576u     /* K3_MAX_CONTEXT */
#define SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT 163840u          /* K3_VOCAB */
#define SPARK_K3_MODEL_RMS_NORM_EPSILON 1e-05f             /* K3_RMS_EPSILON */

/* Deployment and driver policy, not model constants: no config.h source. */
#define SPARK_K3_MODEL_KV_POOL_TOKENS 4194304u             /* deployment sizing, same pool policy as glm52 */
#define SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT 256u         /* pack-time restriction policy */
#define SPARK_K3_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH 256u
#define SPARK_K3_MODEL_END_OF_TEXT_TOKEN_ID 163585u        /* generated_config.h K3_EOS_TOKEN */

/*
 * Mixture of experts: Stable LatentMoE. The routed experts run at their OWN
 * hidden width, not K3_HIDDEN - down_proj 7168 -> 3584, experts at 3584 with
 * intermediate 3072, rms norm, up_proj back to 7168 - and the shared experts
 * are NOT in the latent space: they take the pre-projection hidden at
 * 3072 * 2 = 6144. config.h's LatentMoE block has the full dataflow.
 */
#define SPARK_K3_MODEL_MOE_EXPERT_COUNT 896u               /* K3_EXPERTS */
#define SPARK_K3_MODEL_MOE_TOP_K 16u                       /* K3_TOP_K */
#define SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT 2u          /* K3_SHARED_EXPERTS (was 1) */
#define SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION 3072u    /* K3_EXPERT_INTERMEDIATE (was 2048) */
#define SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION 33792u /* K3_DENSE_INTERMEDIATE (was 18432) */
#define SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR 1.0f      /* K3_ROUTED_SCALE (was 2.5) */
#define SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION 3584u /* K3_ROUTED_EXPERT_HIDDEN; LatentMoE projection width, no pre-audit counterpart */
#define SPARK_K3_MODEL_MOE_NORM_TOPK_PROB 1u               /* router: sigmoid, noaux_tc grouping, renormalised (config.h MoE block) */
#define SPARK_K3_MODEL_MOE_W1_COMPONENT_COUNT 2u           /* gate + up feeding SiTU (K3_SITU_BETA / K3_SITU_LINEAR_BETA) */

/*
 * Hybrid attention layout. Three of every four layers are KDA, the fourth is
 * gated MLA. 0-indexed here: layer % 4 == 3 is MLA, and THE LAST LAYER (92)
 * is an exception - 92 % 4 == 0 would call it KDA. This mirrors config.h's
 * K3_LAYER_IS_LINEAR exactly; the pre-audit 72-layer tree needed no
 * exception. Period and phase match generated_config.h's K3_ATTENTION_PERIOD
 * and K3_GLOBAL_ATTENTION_PHASE.
 */
#define SPARK_K3_MODEL_ATTENTION_PERIOD 4u
#define SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE 3u
#define SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) \
	((((layer_index) % SPARK_K3_MODEL_ATTENTION_PERIOD) != SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE) && \
	 ((layer_index) != (SPARK_K3_MODEL_LAYER_COUNT - 1u)))

/*
 * Kimi Delta Attention per-layer geometry. 96 heads at 128 (was a 64-head
 * guess); per-head keys and queries are L2-normalized IN KERNEL
 * (K3_KDA_QK_L2NORM), beta passes through sigmoid in kernel.
 */
#define SPARK_K3_MODEL_KDA_HEAD_COUNT 96u                  /* K3_KDA_HEADS (was 64) */
#define SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION 128u         /* K3_KDA_KEY_DIM */
#define SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION 128u       /* K3_KDA_VALUE_DIM */
#define SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION 128u         /* forget-gate bottleneck: f_b(f_a(hidden)) is 7168 -> 128 -> 12288 (config.h KDA projections) */
#define SPARK_K3_MODEL_KDA_CHUNK_TOKENS 64u                /* driver chunking, matches the validated kernel */
#define SPARK_K3_MODEL_KDA_QK_L2NORM 1u                    /* K3_KDA_QK_L2NORM */
/*
 * The forget/output gates are FULL RANK: one logit per head PER CHANNEL
 * (96 * 128 = 12288), not the one scalar per head LmDeltaRuleKernel
 * originally read. The kernel has since been widened - linear_attn.cuh reads
 * the gate per channel and LmBoundedDecay applies this model's -5.0 floor -
 * so these macros exist to state the shape, not to work around the kernel.
 */
#define SPARK_K3_MODEL_KDA_FULL_RANK_GATE 1u               /* K3_KDA_FULL_RANK_GATE */
#define SPARK_K3_MODEL_KDA_GATE_LOWER_BOUND -5.0f          /* K3_KDA_GATE_LOWER_BOUND: clamp on the gate */
#define SPARK_K3_MODEL_KDA_A_LOG_SOURCE_HEAD_COUNT 128u    /* K3_KDA_A_LOG_SOURCE_HEADS: checkpoint carries 128 heads, the model runs 96; the loader takes the first-96 slice and refuses other shapes */
/*
 * Each of the q, k and v projections has its OWN short causal convolution
 * with a SiLU, kernel width 4 (K3_KDA_CONV_KERNEL); the three windows live
 * with the recurrent state in the per-sequence slab, priced byte-exact by
 * spark_k3_kv_geometry.h and aliased below.
 */
#define SPARK_K3_MODEL_KDA_CONV_KERNEL 4u                  /* K3_KDA_CONV_KERNEL */
/*
 * Floor on the running (cumulative) log decay inside a chunk - the DRIVER
 * chunk plan's validity condition, distinct from the -5.0 gate clamp above.
 * While the running decay stays above this floor the chunkwise form equals
 * the sequential recurrence; once it engages the two are different models.
 */
#define SPARK_K3_MODEL_KDA_MIN_LOG_DECAY -16.0f
#define SPARK_K3_MODEL_KDA_QK_DIMENSION \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION)
#define SPARK_K3_MODEL_KDA_VALUE_DIMENSION \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION)
/*
 * The recurrent state is FP32 (K3_KDA_STATE_ELEMENT_BYTES): it accumulates
 * over a million tokens without renormalising. The reference stores it
 * TRANSPOSED (transpose_state_layout=True) - the pack/loader owns that flip,
 * the kernel and this header see the untransposed heads x key x value
 * layout. The byte-level slab (state + the three convolution windows) is
 * spark_k3_kv_geometry.h's definition, aliased here so the module has one
 * vocabulary.
 */
#define SPARK_K3_MODEL_KDA_STATE_ELEMENT_BYTES 4u          /* K3_KDA_STATE_ELEMENT_BYTES */
#define SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_HEAD \
	(SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION)
#define SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_LAYER \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_HEAD)
#define SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER
#define SPARK_K3_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER

/*
 * Gated MLA (global attention) per-layer geometry. mla_use_nope and
 * mla_use_output_gate are both true (K3_MLA_USE_NOPE / K3_MLA_OUTPUT_GATE):
 * the nope half carries no rotation, and the attention output is gated -
 * the same gate qwen_3_6 needs, so one kernel serves both.
 */
#define SPARK_K3_MODEL_MLA_HEAD_COUNT 96u                  /* K3_MLA_HEADS (was 64) */
#define SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION 1536u         /* K3_Q_LORA_RANK */
#define SPARK_K3_MODEL_MLA_LATENT_DIMENSION 512u           /* K3_KV_LORA_RANK */
#define SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION 128u     /* K3_QK_NOPE_DIM */
/*
 * NOT ROTATED. K3 is NoPE: modeling_kimi_linear.py sets rotary_emb to None
 * and splits this 64-wide slice out of q and kv only to concatenate it back
 * untouched, the k side broadcast across heads MQA style. Position is
 * carried by KDA's decay. The pre-audit header had this dimension at ZERO,
 * which made the qk head 128 and the cache token 512 - both wrong. The
 * UNROTATED name is the audited one; the ROPE spelling stays for the
 * recovered module under the same convention as spark_k3_kv_geometry.h:
 * the number is the unrotated slice, the name is the machinery's.
 */
#define SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION 64u         /* K3_QK_UNROTATED_DIM */
#define SPARK_K3_MODEL_MLA_ROPE_DIMENSION SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION
#define SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION 128u       /* K3_V_HEAD_DIM */
#define SPARK_K3_MODEL_MLA_USE_NOPE 1u                     /* K3_MLA_USE_NOPE */
#define SPARK_K3_MODEL_MLA_OUTPUT_GATE 1u                  /* K3_MLA_OUTPUT_GATE */
/* 1 / sqrt(qk_nope + unrotated) = 1 / sqrt(128 + 64), read from
 * modeling_kimi_linear.py: self.scaling = q_head_dim ** -0.5
 * (K3_MLA_QK_SCALE; was 1/sqrt(128) at rope-dim zero). */
#define SPARK_K3_MODEL_MLA_QK_SCALE 0.07216878365f
#define SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION \
	(SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION)
#define SPARK_K3_MODEL_MLA_QUERY_B_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION)
#define SPARK_K3_MODEL_MLA_KV_A_DIMENSION \
	(SPARK_K3_MODEL_MLA_LATENT_DIMENSION + SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION)
#define SPARK_K3_MODEL_MLA_KV_B_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * \
	 (SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION))
#define SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS SPARK_K3_MODEL_MLA_KV_A_DIMENSION
#define SPARK_K3_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION)

/*
 * Block Attention Residuals. Blocks of TWELVE layers, not the pre-audit's
 * uniform nine: 93 layers give 7 full blocks plus a PARTIAL final block of
 * 9 layers, the embedding is always b_0, and every sub-layer mixes
 * (completed blocks + running partial sum) with softmax over a learned
 * per-sub-layer pseudo-query against RMS-normalized representations.
 * Applied twice per layer (attention site and MLP site).
 */
#define SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS 12u            /* K3_ATTNRES_BLOCK_SIZE (was 9) */
#define SPARK_K3_MODEL_ATTNRES_BLOCK_COUNT \
	((SPARK_K3_MODEL_LAYER_COUNT + SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS - 1u) / \
	 SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS)                  /* ceil: 8 blocks, one partial final */
/*
 * K3_ATTNRES_MAX_SOURCES: 8 blocks plus the embedding is 9 bank candidates.
 * The running partial is NOT a bank candidate - it travels beside the bank
 * under its own range (config.h's K3_ATTNRES_PARTIAL_BYTES) - and counting
 * it is what made the pre-audit value 10 against the contract's 9
 * (k3_authoritative.json: layer_block_count + embedding_representation_count).
 */
#define SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS 9u
#define SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index) \
	(1u + ((layer_index) / SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS))
#define SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer_index) \
	(((((layer_index) + 1u) % SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS) == 0u) ? 1u : 0u)

/*
 * Quantization: MXFP4 weights, group 32, E8M0 scales (K3_MXFP4_GROUP), and
 * the ignore list matters - attention, shared experts, the dense MLP and
 * lm_head are NOT quantised; only the routed experts are 4-bit. The
 * checkpoint asks for NEITHER MXFP8 nor MXFP4 ACTIVATIONS (both null in its
 * quantization_config): the production path is BF16 activation against a
 * streamed MXFP4 weight, decoded to BF16 registers at the tile. The MXFP8
 * group macro survives only because the recovered module names it; it
 * describes no checkpoint requirement.
 */
#define SPARK_K3_MODEL_MXFP4_GROUP_SIZE 32u                /* K3_MXFP4_GROUP */
#define SPARK_K3_MODEL_MXFP8_GROUP_SIZE 32u                /* pre-audit activation claim; see above */
#define SPARK_K3_MODEL_MXFP4_PAYLOAD_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / 2u)
#define SPARK_K3_MODEL_MXFP4_SCALE_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / (uint64_t)SPARK_K3_MODEL_MXFP4_GROUP_SIZE)

#define SPARK_K3_MODEL_BF16_ELEMENT_BYTES ((uint32_t)sizeof(uint16_t))
#define SPARK_K3_MODEL_HIDDEN_BF16_BYTES \
	(SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES)

/* One geometry source: this shim and spark_k3_kv_geometry.h must agree on
 * the KDA slab's shape or nothing below compiles. */
#if SPARK_K3_MODEL_KDA_HEAD_COUNT != SPARK_K3_KV_KDA_HEADS
#error k3 model shim and kv geometry disagree on the kda head count
#endif
#if SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION != SPARK_K3_KV_KDA_KEY_DIM
#error k3 model shim and kv geometry disagree on the kda key dimension
#endif
#if SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION != SPARK_K3_KV_KDA_VALUE_DIM
#error k3 model shim and kv geometry disagree on the kda value dimension
#endif
#if SPARK_K3_MODEL_KDA_CONV_KERNEL != SPARK_K3_KV_KDA_CONV_KERNEL
#error k3 model shim and kv geometry disagree on the kda convolution kernel
#endif

/* Sanity: geometry the kernels assume. The partial final AttnRes block is
 * legitimate - the check is that the bank candidate count is exactly the
 * blocks plus the embedding, never counting the running partial. */
#if SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS != (SPARK_K3_MODEL_ATTNRES_BLOCK_COUNT + 1u)
#error k3 attnres bank candidates must be the block count plus the embedding
#endif
#if (SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION % 16u) != 0u || \
    (SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION % 64u) != 0u
#error k3 kda head geometry must satisfy the chunk kernel tiling
#endif
#if (SPARK_K3_MODEL_HIDDEN_DIMENSION % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u || \
    (SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u
#error k3 quantized dimensions must be whole mxfp4 groups
#endif

#endif
