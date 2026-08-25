#pragma once

#include <stdint.h>

#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

/*
 * DFlash2 drafter pack: the 5-layer sliding-attention DFlash2 drafter
 * (z-lab/Qwen3.8-27B-DFlash2), packed by tools/qwen38_27b_dspark_stagepack.py into
 * the SAME wire layout as the qwen38_27b target pack
 * (spark_qwen38_27b_stagepack_format.h): magic Q6SP, header 26I2Q, entry 6I4Q. The
 * drafter shares the target's token embedding and lm_head, so this pack carries
 * only the 5-layer backbone + projector + grouped-conv + candidate-selector
 * heads (81 tensors, all BF16).
 *
 * Header field repurposing (mirror the packer): mxfp4_group_size -> block_size
 * (8), mtp_layer_count -> target_tap_count (5). Everything else (hidden 5120,
 * attn 32 Q / 8 KV x 128, ffn 17408, vocab 248320) is literal.
 *
 * DFlash2 vs DSpark geometry delta: query heads 40->32, ffn 10240->17408,
 * block 7->8, taps {4,16,28,40,52}->{5,19,33,47,61}, mask 248077->248070,
 * attention full->sliding (window 2048, is_causal false), confidence head
 * DROPPED, and the Markov W1/W2 slots repurpose to the selector's
 * predecessor/successor codebooks ([248320, 256] — bit-identical shapes, so
 * the pack slot / D2H mirror / byte accounting carry over verbatim).
 */

#define SPARK_QWEN38_27B_DSPARK_LAYER_COUNT 5u
#define SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS 32u
#define SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS 8u
#define SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION 128u
#define SPARK_QWEN38_27B_DSPARK_ATTN_ROPE_DIMENSION 64u
#define SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE 17408u
#define SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE 8u
#define SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT 5u
#define SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK 256u
#define SPARK_QWEN38_27B_DSPARK_SELECTOR_TOP_K 16u
#define SPARK_QWEN38_27B_DSPARK_CONV_KERNEL_SIZE 2u
#define SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE 16u
/* Full-sequence context (z-lab DFlash2): the drafter's K/V context is the
 * projector over EVERY committed position's 5-tap concat (25600 -> 5120), not
 * one position. The taps live in a per-lane rolling ring of capacity 2048 (a
 * power of two so the position index is a single AND); the attention attends
 * over the last CONTEXT_MAX = sliding_window - 1 = 2047 positions + the block,
 * exactly z-lab's RotatingKVCache(max_size = sliding_window - 1). The 2048th
 * ring slot is the wrap-around head and is never read by the context build. */
#define SPARK_QWEN38_27B_DSPARK_TAP_RING_CAPACITY 2048u
#define SPARK_QWEN38_27B_DSPARK_CONTEXT_MAX (SPARK_QWEN38_27B_DSPARK_SLIDING_WINDOW - 1u)
/* The projector input is the 5 taps concatenated per position (fc.weight is
 * [5120, 5*5120]); the ring stores them as one 25600-wide row per position so
 * the projector's N-row launch can read a contiguous row. */
#define SPARK_QWEN38_27B_DSPARK_TAP_ROW_DIMENSION (SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION)
#define SPARK_QWEN38_27B_DSPARK_TAP_RING_CAPACITY_MASK (SPARK_QWEN38_27B_DSPARK_TAP_RING_CAPACITY - 1u)
#define SPARK_QWEN38_27B_DSPARK_SLIDING_WINDOW 2048u
#define SPARK_QWEN38_27B_DSPARK_MASK_TOKEN_ID 248070u

typedef enum SparkQwen38_27bDsparkTensorKind
{
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY = 0,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY = 1,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_VALUE = 2,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_OUTPUT = 3,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY_NORM = 4,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY_NORM = 5,
	SPARK_QWEN38_27B_DSPARK_TENSOR_ATTENTION_NORM = 6,
	SPARK_QWEN38_27B_DSPARK_TENSOR_MLP_NORM = 7,
	SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_GATE = 8,
	SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_UP = 9,
	SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_DOWN = 10,
	SPARK_QWEN38_27B_DSPARK_TENSOR_PROJECTOR = 11,
	SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_PRED = 12,        /* predecessor_codebook [vocab, rank] */
	SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_SUCC = 13,        /* successor_codebook   [vocab, rank] */
	SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_HIDDEN_PROJ = 14, /* hidden_projection.weight [rank, hidden] */
	SPARK_QWEN38_27B_DSPARK_TENSOR_FINAL_NORM = 15,
	SPARK_QWEN38_27B_DSPARK_TENSOR_HIDDEN_NORM = 16,
	SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_BASE = 17, /* attention_conv.base_kernel [2, 2*5120] flattened */
	SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_PROJ = 18, /* attention_conv.kernel_projection.weight [1280, 5120] */
	SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_BASE = 19,  /* mlp_conv.base_kernel [2, 2*5120] flattened */
	SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_PROJ = 20,  /* mlp_conv.kernel_projection.weight [1280, 5120] */
	SPARK_QWEN38_27B_DSPARK_TENSOR_KIND_COUNT = 21
} SparkQwen38_27bDsparkTensorKind;

/* Flattened base_kernel rows/cols: the rank-3 [2,2,5120] (sides, taps,
 * channels) is flattened to [2, 2*5120] = [2, 10240] — rows keep the two
 * sides (prepare/finish), columns concatenate tap*5120+channel. The payload
 * byte order is the safetensors order ([side][tap][channel]) reinterpreted
 * as [side][tap*5120+channel], so no reshape touches the bytes. */
#define SPARK_QWEN38_27B_DSPARK_CONV_BASE_ROWS 2u
#define SPARK_QWEN38_27B_DSPARK_CONV_BASE_COLUMNS (SPARK_QWEN38_27B_DSPARK_CONV_KERNEL_SIZE * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION)
#define SPARK_QWEN38_27B_DSPARK_CONV_PROJ_ROWS (2u * SPARK_QWEN38_27B_DSPARK_CONV_KERNEL_SIZE * (SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION / SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE))

static inline void SparkQwen38_27bDsparkKindShape(
	uint32_t kind,
	uint32_t *rows,
	uint32_t *columns)
{
	switch (kind)
	{
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY:
		*rows = SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS * SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_VALUE:
		*rows = SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS * SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_OUTPUT:
		*rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		*columns = SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS * SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY_NORM:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY_NORM:
		*rows = 1u;
		*columns = SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTENTION_NORM:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_MLP_NORM:
		*rows = 1u;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_GATE:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_UP:
		*rows = SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_DOWN:
		*rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		*columns = SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_PROJECTOR:
		*rows = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		*columns = SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_PRED:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_SUCC:
		*rows = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
		*columns = SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_HIDDEN_PROJ:
		*rows = SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FINAL_NORM:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_HIDDEN_NORM:
		*rows = 1u;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_BASE:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_BASE:
		*rows = SPARK_QWEN38_27B_DSPARK_CONV_BASE_ROWS;
		*columns = SPARK_QWEN38_27B_DSPARK_CONV_BASE_COLUMNS;
		return;
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_PROJ:
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_PROJ:
		*rows = SPARK_QWEN38_27B_DSPARK_CONV_PROJ_ROWS;
		*columns = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		return;
	default:
		*rows = 0u;
		*columns = 0u;
		return;
	}
}
