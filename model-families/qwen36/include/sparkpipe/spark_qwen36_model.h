#pragma once

#include <stdint.h>

/*
 * Qwen 3.6 27B model constants for sparkpipe.
 *
 * Provenance ledger (2026-07-19):
 *   CONFIG       - read directly from Qwen/Qwen3.6-27B config.json (text_config,
 *                  commit 1b559cf), architectures Qwen3_5ForConditionalGeneration,
 *                  model_type qwen3_5. These are facts, not guesses.
 *   MODELING-PIN - the exact functional form must be pinned against the public
 *                  transformers modeling_qwen3_5 source (v4.57.1) before the
 *                  CUDA kernels are written: the GDN recurrence and conv/gating
 *                  order, q/k normalization placement, and the attention output
 *                  gate activation. Unlike K3 there is nothing to guess; the
 *                  reference implementation exists and is the contract.
 *
 * CONFIG architecture: dense 27B hybrid, 64 layers in sixteen [GDN, GDN, GDN,
 * full-attention] periods (layer_types in config, full_attention_interval 4).
 * Gated DeltaNet linear attention with 16 QK heads and 48 V heads at head dim
 * 128 (three value heads share each key head), depthwise causal conv kernel 4,
 * fp32 recurrent state, swish output gate. Full attention is GQA 24 query / 4
 * KV heads at head dim 256 with a learned output gate and partial RoPE over a
 * quarter of the head (64 dims, theta 1e7; mrope sections apply to vision
 * tokens only and text-only serving reduces to standard 1D RoPE). Dense SwiGLU
 * FFN 17408 on every layer, no MoE. Vocabulary 248320 padded, embeddings
 * untied. MTP depth 1 with shared embeddings. Text-only serving: the vision
 * tower in the checkpoint is out of scope for this driver and its tensors are
 * rejected at pack build, not silently dropped at load.
 *
 * Parameter closure (all CONFIG dims): embedding 248320 x 5120 = 1.271B, head
 * (untied) 1.271B, 64 x SwiGLU(3 x 5120 x 17408) = 17.11B, 48 GDN layers x
 * (qkvz 16384 + ba 96 + out 6144, all x 5120, + conv 10240 x 4) = 5.57B, 16
 * attention layers x (q+gate 12288 + kv 2048, x 5120, + o 6144 x 5120) = 1.68B
 * = 26.9B, closing on the advertised 27B. The fused-projection layout inside
 * the checkpoint is a pack-converter concern; the stage pack carries the split
 * tensors named below and the converter splits at build time against the
 * safetensors index.
 */

#define SPARK_QWEN36_MODEL_HIDDEN_DIMENSION 5120u            /* CONFIG hidden_size */
#define SPARK_QWEN36_MODEL_LAYER_COUNT 64u                   /* CONFIG num_hidden_layers */
#define SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS 262144u    /* CONFIG max_position_embeddings */
#define SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT 248320u        /* CONFIG vocab_size (padded) */
#define SPARK_QWEN36_MODEL_RMS_NORM_EPSILON 1e-06f           /* CONFIG rms_norm_eps */
#define SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION 17408u /* CONFIG intermediate_size, SwiGLU (silu) */
#define SPARK_QWEN36_MODEL_BOS_TOKEN_ID 248044u              /* CONFIG bos_token_id (== eos) */
#define SPARK_QWEN36_MODEL_EOS_TOKEN_ID 248044u              /* CONFIG eos_token_id */
#define SPARK_QWEN36_MODEL_TIE_WORD_EMBEDDINGS 0u            /* CONFIG tie_word_embeddings false */
#define SPARK_QWEN36_MODEL_MTP_LAYER_COUNT 1u                /* CONFIG mtp_num_hidden_layers */
#define SPARK_QWEN36_MODEL_MTP_DEDICATED_EMBEDDINGS 0u       /* CONFIG mtp_use_dedicated_embeddings false */

/*
 * Hybrid layout. CONFIG layer_types is sixteen repetitions of [linear,
 * linear, linear, full] and full_attention_interval is 4, so the full
 * attention layer sits last in each period: layer l is GDN iff (l % 4) != 3.
 */
#define SPARK_QWEN36_MODEL_ATTENTION_PERIOD 4u               /* CONFIG full_attention_interval */
#define SPARK_QWEN36_MODEL_FULL_ATTENTION_PHASE 3u           /* CONFIG layer_types order */
#define SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer_index) \
	(((layer_index) % SPARK_QWEN36_MODEL_ATTENTION_PERIOD) != SPARK_QWEN36_MODEL_FULL_ATTENTION_PHASE)
#define SPARK_QWEN36_MODEL_GDN_LAYER_COUNT 48u
#define SPARK_QWEN36_MODEL_FULL_ATTENTION_LAYER_COUNT 16u

/*
 * Gated DeltaNet geometry. Three value heads share each key head's query and
 * key (grouped-value linear attention): the recurrent state is one dk x dv
 * fp32 matrix per VALUE head, updated with its group's normalized k and
 * decayed by that value head's own scalar gate. Decay and beta are per value
 * head scalars (the in_proj_ba pair), which is the scalar special case of the
 * per-channel decay the K3 chunk kernel already implements. The depthwise
 * causal conv (kernel 4, silu) runs over the concatenated q|k|v channels
 * before heads are formed; its per-lane last-3-columns tail is recurrent
 * state carried across dispatches exactly like the delta state.
 */
#define SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT 16u            /* CONFIG linear_num_key_heads */
#define SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT 48u          /* CONFIG linear_num_value_heads */
#define SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION 128u       /* CONFIG linear_key_head_dim */
#define SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION 128u     /* CONFIG linear_value_head_dim */
#define SPARK_QWEN36_MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD \
	(SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN36_MODEL_GDN_CONV_KERNEL 4u                /* CONFIG linear_conv_kernel_dim */
#define SPARK_QWEN36_MODEL_GDN_OUTPUT_GATE_SWISH 1u          /* CONFIG output_gate_type "swish" */
#define SPARK_QWEN36_MODEL_GDN_QK_DIMENSION \
	(SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT * SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION)
#define SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION \
	(SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION)
#define SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS \
	((2u * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION) + SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION)
#define SPARK_QWEN36_MODEL_GDN_CONV_TAIL_COLUMNS (SPARK_QWEN36_MODEL_GDN_CONV_KERNEL - 1u)
#define SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS 64u              /* driver chunk width, matches the K3 kernel plan */
#define SPARK_QWEN36_MODEL_GDN_STATE_ELEMENTS_PER_HEAD \
	(SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION)
#define SPARK_QWEN36_MODEL_GDN_STATE_ELEMENTS_PER_LAYER \
	((uint64_t)SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_GDN_STATE_ELEMENTS_PER_HEAD)
#define SPARK_QWEN36_MODEL_GDN_CONV_TAIL_ELEMENTS_PER_LAYER \
	((uint64_t)SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN36_MODEL_GDN_CONV_TAIL_COLUMNS)

/*
 * Full attention geometry. GQA with six query heads per KV head at head dim
 * 256. RoPE covers the first quarter of the head (partial_rotary_factor
 * 0.25): 64 dims rotated, 192 pass through, theta 1e7. attn_output_gate is a
 * learned per-element gate on the attention output before the o projection
 * (MODELING-PIN: activation of that gate). Query and key head RMSNorm follow
 * the Qwen3 lineage (MODELING-PIN: weight shape and placement). The KV cache
 * stores K and V post-RoPE at full head width, paged like the K3 latent cache.
 */
#define SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT 24u         /* CONFIG num_attention_heads */
#define SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT 4u             /* CONFIG num_key_value_heads */
#define SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION 256u          /* CONFIG head_dim */
#define SPARK_QWEN36_MODEL_ATTN_QUERIES_PER_KV_HEAD \
	(SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT)
#define SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION 64u           /* CONFIG partial_rotary_factor 0.25 x 256 */
#define SPARK_QWEN36_MODEL_ATTN_ROPE_THETA 10000000.0f       /* CONFIG rope_theta */
#define SPARK_QWEN36_MODEL_ATTN_OUTPUT_GATE 1u               /* CONFIG attn_output_gate true */
#define SPARK_QWEN36_MODEL_ATTN_QUERY_DIMENSION \
	(SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION \
	(SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS \
	(2u * SPARK_QWEN36_MODEL_ATTN_KV_DIMENSION)
#define SPARK_QWEN36_MODEL_ATTN_QK_SCALE 0.0625f             /* 1/sqrt(256) */

#define SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES 2u
#define SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES \
	(SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES)

/* The layer map, head grouping and RoPE split must be internally consistent. */
