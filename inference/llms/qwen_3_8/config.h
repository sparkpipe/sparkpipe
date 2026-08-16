#pragma once
// Qwen 3.8 Max (Qwen3.8-2.4T-A95B). Shapes and constants only.
//
// AUDITED against the pinned vendor FP8 checkpoint
// (Qwen/Qwen3.8-2.4T-A95B-FP8, revision d2dc3565, index and config
// hash-verified; model_contracts/qwen38_authoritative.json pins it).
// Hidden 8192, 92 layers, vocab 248320, ctx 262144 native. The layer
// layout is 23 x (3 x gated DeltaNet -> 1 x gated attention), the same
// period-4 phase-3 arrangement as Qwen 3.6 but with a routed MoE on
// EVERY layer: 512 experts, top-10, intermediate 2048, plus one shared
// expert weighted by a learned per-channel gate.
//
// PRECISION: the vendor FP8 release ships routed experts as block-128
// FP8 (E4M3 payload, per-block F32 weight_scale_inv) and everything else
// BF16; the driver consumes experts as LmFp8 (kScaleGroup 128, block-128
// F32 scales) and the BF16 spine as BF16. This is the quality-first
// policy: vendor FP8 experts as shipped, no MXFP4 requantization.
//
// The GDN recurrence is FP32 state (gated DeltaNet compounds per token;
// a bf16-state variant would be a separate, differently-named kernel).
#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

#define QWEN38_HIDDEN 8192u                    /* CONFIG hidden_size */
#define QWEN38_LAYERS 92u                      /* CONFIG num_hidden_layers */
#define QWEN38_VOCAB 248320u                   /* CONFIG vocab_size, padded */
#define QWEN38_RMS_EPSILON 1e-06f              /* CONFIG rms_norm_eps */

#define QWEN38_ATTENTION_PERIOD 4u             /* 23 x (3 GDN -> 1 attn) */
#define QWEN38_FULL_PHASE 3u                   /* CONFIG layer_types order */
#define QWEN38_LAYER_IS_LINEAR(layer) (((layer) % QWEN38_ATTENTION_PERIOD) != QWEN38_FULL_PHASE)

// Full attention, one layer in four: 64 query heads over 4 KV heads at
// head dim 256, rope on the first 64, gated output, per-head q/k norms.
#define QWEN38_ATTN_HEADS 64u                  /* CONFIG num_attention_heads */
#define QWEN38_KV_HEADS 4u                     /* CONFIG num_key_value_heads */
#define QWEN38_HEAD_DIM 256u                   /* CONFIG head_dim */
#define QWEN38_ROPE_DIM 64u                    /* CONFIG partial_rotary_factor 0.25 x 256 */
#define QWEN38_NOPE_DIM (QWEN38_HEAD_DIM - QWEN38_ROPE_DIM)
#define QWEN38_ROPE_THETA 10000000.0f          /* CONFIG rope_theta */
#define QWEN38_QK_SCALE 0.0625f                /* 1 / sqrt(256) */
#define QWEN38_ATTN_OUTPUT_GATE 1u             /* CONFIG attn_output_gate */
// query|gate + key + value fused projection rows.
#define QWEN38_QKV_DIM (((2u * QWEN38_ATTN_HEADS) + (2u * QWEN38_KV_HEADS)) * QWEN38_HEAD_DIM)

#define QWEN38_GDN_KEY_HEADS 16u               /* CONFIG linear_num_key_heads */
#define QWEN38_GDN_VALUE_HEADS 128u            /* CONFIG linear_num_value_heads */
#define QWEN38_GDN_KEY_DIM 128u                /* CONFIG linear_key_head_dim */
#define QWEN38_GDN_VALUE_DIM 128u              /* CONFIG linear_value_head_dim */
#define QWEN38_GDN_CONV_KERNEL 4u              /* CONFIG linear_conv_kernel_dim */

// One fp32 delta-rule state per VALUE head (16x128x128 floats = 2 MiB per
// sequence per GDN layer), plus the short causal conv window. Both are
// per-sequence and never grow.
#define QWEN38_GDN_STATE_ELEMENT_BYTES 4u
#define QWEN38_GDN_STATE_BYTES 	((QWEN38_GDN_VALUE_HEADS * QWEN38_GDN_KEY_DIM * QWEN38_GDN_VALUE_DIM 		* QWEN38_GDN_STATE_ELEMENT_BYTES) 	 + (QWEN38_GDN_QKV_DIM * QWEN38_GDN_CONV_KERNEL * 2u))

// Routed MoE on every layer.
#define QWEN38_EXPERTS 512u                    /* CONFIG num_experts */
#define QWEN38_TOP_K 10u                       /* CONFIG num_experts_per_tok */
#define QWEN38_EXPERT_INTERMEDIATE 2048u       /* CONFIG expert_intermediate_size */
#define QWEN38_SHARED_EXPERTS 1u               /* CONFIG n_shared_experts */

#define QWEN38_MTP_LAYERS 1u                   /* CONFIG mtp_num_hidden_layers */
#define QWEN38_KV_BITS 16u
#define QWEN38_KV_PAGE_SLOTS 64u

// -- layer kinds --
// 23 x (3 x gated DeltaNet -> 1 x gated full attention), MoE on every layer.
#define QWEN38_LAYER_KIND(layer) 	(QWEN38_LAYER_IS_LINEAR(layer) ? LM_LAYER_RECURRENT : LM_LAYER_FULL)
