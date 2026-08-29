#pragma once

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"

/*
 * K3DS: the DSpark drafter wire pack for the k3 family, written by
 * tools/k3_dspark_stagepack.py and bound by the drafter-pack path in
 * spark_k3_pack_load.c. The wire discipline is the qwen38_27b drafter
 * pack's (spark_qwen38_27b_dspark_format.h): magic + version, a fixed
 * little-endian header, 56-byte entries (6I4Q), 256-aligned payloads -
 * but the header EXTENDS the 26I2Q Q6SP core by 16 U32 fields, because
 * the admitted DSpark sources vary per release and the pack must be
 * self-describing. The kimi-k3 drafter SHIPS its own embed_tokens and
 * lm_head (DFlash2 shares the target's), and carries a Markov bias head
 * and a confidence head instead of DFlash2's candidate selector.
 *
 * Pinned source (2026-08-29, DOWNLOAD-RECEIPT.json on warm): redhatai
 * DSparkDraftModel - hidden 7168, 5 layers, 96 Q / 16 KV heads x 64,
 * FFN 14336, vocab 163840, block 8, taps {24,48,72,88,92}, markov_rank
 * 256, mask 163837, window 2048, rope theta 10000 (default), confidence
 * head over hidden+markov (7424 wide), 64 tensors, all BF16. The BIND
 * reads the geometry FROM THE PACK HEADER and checks it against these
 * constants: a pack that disagrees fails the bind naming the field (the
 * supports() -> WHY rule), it is never silently loaded.
 *
 * NOT the original moonshotai release constants: inference/llms/kimi_k3/
 * dspark.h pins that first DSpark release (block 7, taps {7,23,51,67,83},
 * mask 163824, 64 Q heads). The admitted redhatai source is a different
 * release of the same method - block 8, different taps, 96 Q heads - and
 * the two pins deliberately do NOT share identifiers. RadixArk (block 7)
 * and inferact (different config schema) are NOT covered by these
 * constants; their packs fail the bind naming the field until their
 * geometry is source-verified and pinned the same way.
 *
 * Kind numbering 0..16 mirrors SparkQwen38_27bDsparkTensorKind so the two
 * drafter readers stay legible side by side: 11 is the target-tap
 * projector (fc.weight) in both; slots 12/13 - which DFlash2 repurposed
 * for its selector codebooks - carry the Markov W1/W2 weights whose
 * [vocab, rank] shapes are exactly what DFlash2 repurposed them for; 14
 * is reserved (no DSpark counterpart). 17..20 are the tensors a DSpark
 * drafter ships and a DFlash2 drafter does not.
 */

#define SPARK_K3_DSPARK_MAGIC 0x5344334Bu                /* 'K3DS' */
#define SPARK_K3_DSPARK_FORMAT_VERSION 3u                /* Q6SP v3 wire discipline */
#define SPARK_K3_DSPARK_CORE_HEADER_BYTES 120u           /* the 26I2Q core */
#define SPARK_K3_DSPARK_EXT_HEADER_BYTES 64u             /* 16 U32 DSpark fields */
#define SPARK_K3_DSPARK_HEADER_BYTES \
	(SPARK_K3_DSPARK_CORE_HEADER_BYTES + SPARK_K3_DSPARK_EXT_HEADER_BYTES)
#define SPARK_K3_DSPARK_ENTRY_BYTES 56u
#define SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT 256u
#define SPARK_K3_DSPARK_WEIGHT_BF16 0u

#define SPARK_K3_DSPARK_LAYER_COUNT 5u
#define SPARK_K3_DSPARK_ATTN_QUERY_HEADS 96u
#define SPARK_K3_DSPARK_ATTN_KV_HEADS 16u
#define SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION 64u
#define SPARK_K3_DSPARK_ATTN_ROPE_DIMENSION 64u
#define SPARK_K3_DSPARK_FFN_INTERMEDIATE 14336u
#define SPARK_K3_DSPARK_VOCAB 163840u
#define SPARK_K3_DSPARK_BLOCK_SIZE 8u
#define SPARK_K3_DSPARK_TARGET_TAP_COUNT 5u
#define SPARK_K3_DSPARK_TARGET_TAP_LAYER_0 24u
#define SPARK_K3_DSPARK_TARGET_TAP_LAYER_1 48u
#define SPARK_K3_DSPARK_TARGET_TAP_LAYER_2 72u
#define SPARK_K3_DSPARK_TARGET_TAP_LAYER_3 88u
#define SPARK_K3_DSPARK_TARGET_TAP_LAYER_4 92u
#define SPARK_K3_DSPARK_MARKOV_RANK 256u
#define SPARK_K3_DSPARK_MASK_TOKEN_ID 163837u
#define SPARK_K3_DSPARK_SLIDING_WINDOW 2048u
#define SPARK_K3_DSPARK_ROPE_THETA 10000.0f
/* confidence head input: hidden CONCATENATED WITH the markov latent
 * (confidence_head_with_markov) - sizing it at the hidden alone reads
 * the wrong features */
#define SPARK_K3_DSPARK_CONFIDENCE_INPUT_DIMENSION \
	(K3_HIDDEN + SPARK_K3_DSPARK_MARKOV_RANK)
/* the block emits the anchor (the just-committed token) plus block-1
 * drafts; the verify window walks the drafts */
#define SPARK_K3_DSPARK_MAX_DRAFT_TOKEN_COUNT (SPARK_K3_DSPARK_BLOCK_SIZE - 1u)

/* ext-header U32 offsets (past the 120-byte core) */
#define SPARK_K3_DSPARK_EXT_TAP_LAYER_0 0u
#define SPARK_K3_DSPARK_EXT_MARKOV_RANK 5u
#define SPARK_K3_DSPARK_EXT_MASK_TOKEN_ID 6u
#define SPARK_K3_DSPARK_EXT_SLIDING_WINDOW 7u
#define SPARK_K3_DSPARK_EXT_FLAGS 8u
#define SPARK_K3_DSPARK_EXT_ROPE_THETA_MILLI 9u
#define SPARK_K3_DSPARK_EXT_CONFIDENCE_INPUT_DIM 10u

/* flags: which optional heads the pack carries */
#define SPARK_K3_DSPARK_FLAG_EMBED UINT32_C(0x1)
#define SPARK_K3_DSPARK_FLAG_LM_HEAD UINT32_C(0x2)
#define SPARK_K3_DSPARK_FLAG_CONFIDENCE UINT32_C(0x4)
#define SPARK_K3_DSPARK_FLAG_CONFIDENCE_WITH_MARKOV UINT32_C(0x8)
#define SPARK_K3_DSPARK_FLAG_SLIDING_ATTENTION UINT32_C(0x10)
#define SPARK_K3_DSPARK_FLAGS_REQUIRED \
	(SPARK_K3_DSPARK_FLAG_EMBED | SPARK_K3_DSPARK_FLAG_LM_HEAD | \
	 SPARK_K3_DSPARK_FLAG_CONFIDENCE | \
	 SPARK_K3_DSPARK_FLAG_CONFIDENCE_WITH_MARKOV)

typedef enum SparkK3DsparkTensorKind
{
	SPARK_K3_DSPARK_TENSOR_ATTN_QUERY = 0,
	SPARK_K3_DSPARK_TENSOR_ATTN_KEY = 1,
	SPARK_K3_DSPARK_TENSOR_ATTN_VALUE = 2,
	SPARK_K3_DSPARK_TENSOR_ATTN_OUTPUT = 3,
	SPARK_K3_DSPARK_TENSOR_ATTN_QUERY_NORM = 4,
	SPARK_K3_DSPARK_TENSOR_ATTN_KEY_NORM = 5,
	SPARK_K3_DSPARK_TENSOR_ATTENTION_NORM = 6,
	SPARK_K3_DSPARK_TENSOR_MLP_NORM = 7,
	SPARK_K3_DSPARK_TENSOR_FFN_GATE = 8,
	SPARK_K3_DSPARK_TENSOR_FFN_UP = 9,
	SPARK_K3_DSPARK_TENSOR_FFN_DOWN = 10,
	SPARK_K3_DSPARK_TENSOR_PROJECTOR = 11,       /* fc.weight [hidden, taps*hidden] */
	SPARK_K3_DSPARK_TENSOR_MARKOV_W1 = 12,       /* markov_head.markov_w1 [vocab, rank] */
	SPARK_K3_DSPARK_TENSOR_MARKOV_W2 = 13,       /* markov_head.markov_w2 [vocab, rank] */
	SPARK_K3_DSPARK_TENSOR_RESERVED_14 = 14,     /* DFlash2 selector hidden projection slot */
	SPARK_K3_DSPARK_TENSOR_FINAL_NORM = 15,
	SPARK_K3_DSPARK_TENSOR_HIDDEN_NORM = 16,
	SPARK_K3_DSPARK_TENSOR_EMBED = 17,           /* embed_tokens.weight [vocab, hidden] */
	SPARK_K3_DSPARK_TENSOR_LM_HEAD = 18,         /* lm_head.weight [vocab, hidden] */
	SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT = 19, /* [1, hidden+markov] */
	SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS = 20,   /* [1, 1] */
	SPARK_K3_DSPARK_TENSOR_KIND_COUNT = 21
} SparkK3DsparkTensorKind;

/* redhatai inventory: 5 layers x 11 per-layer kinds + 9 globals */
#define SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT 11u
#define SPARK_K3_DSPARK_GLOBAL_KIND_COUNT 9u
#define SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT \
	(SPARK_K3_DSPARK_LAYER_COUNT * SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT + \
	 SPARK_K3_DSPARK_GLOBAL_KIND_COUNT)

static inline void SparkK3DsparkKindShape(
	uint32_t kind,
	uint32_t *rows,
	uint32_t *columns)
{
	switch (kind)
	{
	case SPARK_K3_DSPARK_TENSOR_ATTN_QUERY:
		*rows = SPARK_K3_DSPARK_ATTN_QUERY_HEADS * SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_ATTN_KEY:
	case SPARK_K3_DSPARK_TENSOR_ATTN_VALUE:
		*rows = SPARK_K3_DSPARK_ATTN_KV_HEADS * SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_ATTN_OUTPUT:
		*rows = K3_HIDDEN;
		*columns = SPARK_K3_DSPARK_ATTN_QUERY_HEADS * SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION;
		return;
	case SPARK_K3_DSPARK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_K3_DSPARK_TENSOR_ATTN_KEY_NORM:
		*rows = 1u;
		*columns = SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION;
		return;
	case SPARK_K3_DSPARK_TENSOR_ATTENTION_NORM:
	case SPARK_K3_DSPARK_TENSOR_MLP_NORM:
		*rows = 1u;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_FFN_GATE:
	case SPARK_K3_DSPARK_TENSOR_FFN_UP:
		*rows = SPARK_K3_DSPARK_FFN_INTERMEDIATE;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_FFN_DOWN:
		*rows = K3_HIDDEN;
		*columns = SPARK_K3_DSPARK_FFN_INTERMEDIATE;
		return;
	case SPARK_K3_DSPARK_TENSOR_PROJECTOR:
		*rows = K3_HIDDEN;
		*columns = SPARK_K3_DSPARK_TARGET_TAP_COUNT * K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_MARKOV_W1:
	case SPARK_K3_DSPARK_TENSOR_MARKOV_W2:
		*rows = SPARK_K3_DSPARK_VOCAB;
		*columns = SPARK_K3_DSPARK_MARKOV_RANK;
		return;
	case SPARK_K3_DSPARK_TENSOR_FINAL_NORM:
	case SPARK_K3_DSPARK_TENSOR_HIDDEN_NORM:
		*rows = 1u;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_EMBED:
	case SPARK_K3_DSPARK_TENSOR_LM_HEAD:
		*rows = SPARK_K3_DSPARK_VOCAB;
		*columns = K3_HIDDEN;
		return;
	case SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT:
		*rows = 1u;
		*columns = SPARK_K3_DSPARK_CONFIDENCE_INPUT_DIMENSION;
		return;
	case SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS:
		*rows = 1u;
		*columns = 1u;
		return;
	default:
		*rows = 0u;
		*columns = 0u;
		return;
	}
}
