#pragma once

#include <stdint.h>

/*
 * Kimi K3 model constants for sparkpipe.
 *
 * Provenance ledger (2026-07-17, weights + technical report land 2026-07-27):
 *   DISCLOSED  - stated by Moonshot (kimi.com/blog/kimi-k3, platform.kimi.ai,
 *                MoonshotAI/Attention-Residuals, arXiv 2510.26692 Kimi Linear,
 *                arXiv 2603.15031 AttnRes).
 *   GUESS      - chosen by us, constrained by disclosed totals; every GUESS is
 *                a single #define and the driver derives everything else, so a
 *                wrong guess is a one-line fix when the report lands.
 *
 * DISCLOSED architecture: Kimi Delta Attention (KDA) hybrid linear attention,
 * Attention Residuals (Block AttnRes), Stable LatentMoE activating 16 of 896
 * experts plus shared experts, Gated MLA for the global-attention layers,
 * SiTU (Sigmoid Tanh Unit) activations, MXFP4 weights with MXFP8 activations
 * trained in from SFT, 1M-token context, ~2.8T total parameters, community
 * consensus ~50B active per token.
 *
 * GUESS anchor: 7168 hidden x 2048 moe-intermediate x 3 matrices x 896 experts
 * x 71 routed layers = 2.801T routed parameters, and 16 active experts over 71
 * routed layers = 50.03B active routed parameters -- both land exactly on the
 * disclosed 2.8T / community A50B figures, and 7168/2048/64-head is the K2
 * lineage blueprint Moonshot has kept stable across K2..K2.7.
 */

#define SPARK_K3_MODEL_HIDDEN_DIMENSION 7168u              /* GUESS (K2 lineage, closes the 2.8T identity) */
#define SPARK_K3_MODEL_LAYER_COUNT 72u                     /* GUESS (71 routed + 1 leading dense) */
#define SPARK_K3_MODEL_FIRST_ROUTED_LAYER 1u               /* GUESS (K2 uses one leading dense layer) */
#define SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS 1048576u     /* DISCLOSED 1M context */
#define SPARK_K3_MODEL_KV_POOL_TOKENS 4194304u             /* deployment sizing, same pool policy as glm52 */
#define SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT 163840u          /* GUESS (K2 tokenizer family) */
#define SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT 256u
#define SPARK_K3_MODEL_RMS_NORM_EPSILON 1e-05f             /* GUESS (K2 value) */
#define SPARK_K3_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH 256u

/* Mixture of experts: Stable LatentMoE. */
#define SPARK_K3_MODEL_MOE_EXPERT_COUNT 896u               /* DISCLOSED */
#define SPARK_K3_MODEL_MOE_TOP_K 16u                       /* DISCLOSED */
#define SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT 1u          /* GUESS (launch coverage says "alongside shared experts"; K2 uses 1) */
#define SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION 2048u    /* GUESS (K2 value, closes the 2.8T identity) */
#define SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION 18432u /* GUESS (K2 value) */
#define SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR 2.5f      /* GUESS (K2 value; Quantile Balancing is train-time only) */
#define SPARK_K3_MODEL_MOE_NORM_TOPK_PROB 1u               /* GUESS (K2 normalizes top-k sigmoid scores) */
#define SPARK_K3_MODEL_MOE_W1_COMPONENT_COUNT 2u           /* gate + up feeding SiTU */

/*
 * Hybrid attention layout. Kimi Linear (DISCLOSED lineage of KDA) interleaves
 * three KDA layers with one full-attention layer; K3 keeps "hybrid linear
 * attention" language, so period 4 with the global layer last in each period.
 * A layer index l is a KDA layer iff (l % SPARK_K3_MODEL_ATTENTION_PERIOD) !=
 * SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE.
 */
#define SPARK_K3_MODEL_ATTENTION_PERIOD 4u                 /* GUESS (Kimi Linear 3:1) */
#define SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE 3u           /* GUESS (Kimi Linear puts MLA last in the period) */
#define SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) \
	(((layer_index) % SPARK_K3_MODEL_ATTENTION_PERIOD) != SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE)

/*
 * Kimi Delta Attention per-layer geometry. dk = dv = 128 matches the Kimi
 * Linear head shape and the validated sparkpipe KDA kernel geometry; per-head
 * keys and queries are L2-normalized (Kimi Linear engineering contract, and
 * the precondition of the chunkwise kernel: non-expansive iff b*||k||^2 <= 2).
 */
#define SPARK_K3_MODEL_KDA_HEAD_COUNT 64u                  /* GUESS */
#define SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION 128u         /* GUESS (Kimi Linear head dim) */
#define SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION 128u       /* GUESS (Kimi Linear head dim) */
#define SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION 128u         /* GUESS (low-rank gate/beta/output-gate projections) */
#define SPARK_K3_MODEL_KDA_CHUNK_TOKENS 64u                /* driver chunking, matches the validated kernel */
/*
 * Floor on the running (cumulative) log decay inside a chunk. It bounds the
 * exp(-running) factor the chunk plan carries in bf16, and it is the plan's
 * validity condition: while the running decay stays above this floor the
 * chunkwise form equals the sequential recurrence (cpu oracle: 3.1e-6), and
 * once it engages the two are different models (cpu oracle: 3.8e+00), because
 * the gram coupling exp(running_c - running_r) collapses toward one. At this
 * floor and a 64 token chunk the plan needs a mean log decay above -0.25 per
 * token. K3's decay parameterization is a GUESS; if it proves aggressive, the
 * chunk width or this floor changes, not the tolerance.
 */
#define SPARK_K3_MODEL_KDA_MIN_LOG_DECAY -16.0f
#define SPARK_K3_MODEL_KDA_QK_DIMENSION \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION)
#define SPARK_K3_MODEL_KDA_VALUE_DIMENSION \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION)
#define SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_HEAD \
	(SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION)
#define SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_LAYER \
	(SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_HEAD)

/*
 * Gated MLA (global attention) per-layer geometry. Kimi Linear runs its MLA
 * layers without positional encoding (NoPE) and lets KDA carry position; K3
 * keeps that here, so the rope dimension is zero and the compressed cache
 * token is exactly the kv latent. "Gated" is a learned per-head sigmoid gate
 * on the attention output before the output projection (the launch blog pairs
 * "Gated MLA" with "attention selectivity"); the gate projection is
 * hidden -> head_count.
 */
#define SPARK_K3_MODEL_MLA_HEAD_COUNT 64u                  /* GUESS (K2 value) */
#define SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION 1536u         /* GUESS (K2 q lora rank) */
#define SPARK_K3_MODEL_MLA_LATENT_DIMENSION 512u           /* GUESS (K2 kv lora rank) */
#define SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION 128u     /* GUESS (K2 value) */
#define SPARK_K3_MODEL_MLA_ROPE_DIMENSION 0u               /* GUESS (Kimi Linear NoPE full attention) */
#define SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION 128u       /* GUESS (K2 value) */
#define SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION \
	(SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_ROPE_DIMENSION)
#define SPARK_K3_MODEL_MLA_QUERY_B_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_QK_HEAD_DIMENSION)
#define SPARK_K3_MODEL_MLA_KV_A_DIMENSION \
	(SPARK_K3_MODEL_MLA_LATENT_DIMENSION + SPARK_K3_MODEL_MLA_ROPE_DIMENSION)
#define SPARK_K3_MODEL_MLA_KV_B_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * \
	 (SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION + SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION))
#define SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS SPARK_K3_MODEL_MLA_KV_A_DIMENSION
#define SPARK_K3_MODEL_MLA_QK_SCALE 0.08838834764831845f   /* 1/sqrt(128); rope dim is zero */
#define SPARK_K3_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION \
	(SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION)

/*
 * Block Attention Residuals (DISCLOSED mechanism, MoonshotAI/Attention-Residuals).
 * Sub-layer granularity: each transformer layer contributes ATTN and MLP, a
 * block closes every SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS layers, the token
 * embedding is block representation zero, and every sub-layer mixes
 * (completed blocks + running partial sum) with softmax over a learned
 * per-sub-layer pseudo-query against RMS-normalized representations.
 * 72 layers / 9 layers per block = 8 blocks, the paper's recommended count.
 */
#define SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS 9u             /* GUESS (yields the paper's ~8 blocks at 72 layers) */
#define SPARK_K3_MODEL_ATTNRES_BLOCK_COUNT \
	(SPARK_K3_MODEL_LAYER_COUNT / SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS)
#define SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS \
	(SPARK_K3_MODEL_ATTNRES_BLOCK_COUNT + 2u)
#define SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index) \
	(1u + ((layer_index) / SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS))
#define SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer_index) \
	(((((layer_index) + 1u) % SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS) == 0u) ? 1u : 0u)

/*
 * Quantization. DISCLOSED: MXFP4 weights (E2M1 payload, E8M0 scale per group
 * of 32) with MXFP8 activations (E4M3 payload, E8M0 scale per group of 32),
 * quantization-aware from SFT. bf16 weight fallbacks remain loadable for
 * bring-up and parity work.
 */
#define SPARK_K3_MODEL_MXFP4_GROUP_SIZE 32u                /* DISCLOSED */
#define SPARK_K3_MODEL_MXFP8_GROUP_SIZE 32u                /* DISCLOSED */
#define SPARK_K3_MODEL_MXFP4_PAYLOAD_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / 2u)
#define SPARK_K3_MODEL_MXFP4_SCALE_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / (uint64_t)SPARK_K3_MODEL_MXFP4_GROUP_SIZE)

/* Token ids: GUESS pending the released tokenizer; overridable at pack build. */
#define SPARK_K3_MODEL_END_OF_TEXT_TOKEN_ID 163585u
#define SPARK_K3_MODEL_BF16_ELEMENT_BYTES ((uint32_t)sizeof(uint16_t))
#define SPARK_K3_MODEL_HIDDEN_BF16_BYTES \
	(SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES)

/* Sanity: geometry the kernels assume. */
#if (SPARK_K3_MODEL_LAYER_COUNT % SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS) != 0u
#error k3 layer count must be a whole number of attnres blocks
#endif
#if (SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION % 16u) != 0u || \
    (SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION % 64u) != 0u
#error k3 kda head geometry must satisfy the chunk kernel tiling
#endif
#if (SPARK_K3_MODEL_HIDDEN_DIMENSION % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u || \
    (SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u
#error k3 quantized dimensions must be whole mxfp4 groups
#endif
