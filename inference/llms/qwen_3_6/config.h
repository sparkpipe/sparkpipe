#pragma once
// Qwen 3.6. Shapes and constants only.
//
// AUDITED against Qwen/Qwen3.6-27B, 2026-07-27. Every constant below matches the
// published card: hidden 5120, 64 layers, vocab 248320 padded, FFN 17408, gated
// DeltaNet 48 value heads over 16 QK at head dim 128, gated attention 24 query
// over 4 KV at head dim 256, rope dimension 64, rope theta 1e7, rms eps 1e-6.
// The layer layout is 16 x (3 x DeltaNet -> 1 x Attention), which is what
// QWEN36_LAYER_IS_LINEAR computes: layers 0,1,2 linear and 3 full, repeating.
//
// WHICH CHECKPOINT: the 27B, which has a dense FFN. The 35B-A3B is a different
// model with the same name prefix - hidden 2048, 40 layers, 256 experts at
// intermediate 512, 16 query heads over 2 KV, 32 value heads. layer.cuh
// implements the dense path and would be wrong for A3B, so the variant is named
// here rather than left to whoever reads the weights.
//
// ONE KNOWN GAP, real and not implemented:
//   the reference sets mrope_interleaved with sections [11,11,10]. LmRopePerHead
//   applies plain rope. For text-only decode the sections degenerate to the
//   standard rotation, so this is correct until an image or video enters.
//
// Two former gaps are closed. The attention output gate is applied:
// attn_output_gate is true in the reference config, the query projection's
// fused per-head gate half is split out by LmSplitQueryGateKernel and
// LmOutputGateKernel applies sigmoid(gate) to the attended values before the
// output projection. And the GDN forget and write gates have a producer:
// beta and decay arrive as separate 48-row projections (the checkpoint's
// fused in_proj_ba, split), and LmGdnGateKernel computes
// beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias) per value
// head from the checkpoint's A_log and dt_bias tensors.
//
// Gated DeltaNet on 48 of 64 layers, full attention on the other 16, in a fixed
// period. Same shape as K3 - a recurrent state for most layers and a KV cache
// for a few - reached from a different direction, which is the argument that
// this is an architecture class rather than one vendor's choice.
#include <stdint.h>
#include "inference/kernels/layer_kind.cuh"

#define QWEN36_HIDDEN 5120u                    /* CONFIG hidden_size */
#define QWEN36_LAYERS 64u                      /* CONFIG num_hidden_layers */
#define QWEN36_VOCAB 248320u                   /* CONFIG vocab_size, padded */
#define QWEN36_RMS_EPSILON 1e-06f              /* CONFIG rms_norm_eps */
#define QWEN36_FFN_INTERMEDIATE 17408u         /* CONFIG intermediate_size */

#define QWEN36_ATTENTION_PERIOD 4u             /* CONFIG full_attention_interval */
#define QWEN36_FULL_PHASE 3u                   /* CONFIG layer_types order */
#define QWEN36_LAYER_IS_LINEAR(layer) (((layer) % QWEN36_ATTENTION_PERIOD) != QWEN36_FULL_PHASE)

// Full attention, one layer in four. These were not here: unity.cu instantiated
// the KV geometry as 8 heads x 128 dims and the decode kernel as latent 128 plus
// rope 64, none of which is this model. The checkpoint says 24 query heads over
// 4 KV heads at head dim 256, rope on the first 64. The byte count per token is
// why it went unnoticed - 8 x 128 and 4 x 256 are both 1024 elements, so the
// pool sized correctly while the layout underneath it was another model's.
#define QWEN36_ATTN_HEADS 24u                  /* CONFIG num_attention_heads */
#define QWEN36_KV_HEADS 4u                     /* CONFIG num_key_value_heads */
#define QWEN36_HEAD_DIM 256u                   /* CONFIG head_dim */
#define QWEN36_ROPE_DIM 64u                    /* CONFIG partial_rotary_factor 0.25 x 256 */
#define QWEN36_NOPE_DIM (QWEN36_HEAD_DIM - QWEN36_ROPE_DIM)
#define QWEN36_ROPE_THETA 10000000.0f          /* CONFIG rope_theta */
#define QWEN36_QK_SCALE 0.0625f                /* 1 / sqrt(256) */
#define QWEN36_ATTN_OUTPUT_GATE 1u             /* CONFIG attn_output_gate */
// The fused projection is query|gate + key + value: the gate doubles the
// query section (256 query rows then 256 gate rows per head).
#define QWEN36_QKV_DIM (((2u * QWEN36_ATTN_HEADS) + (2u * QWEN36_KV_HEADS)) * QWEN36_HEAD_DIM)

#define QWEN36_GDN_KEY_HEADS 16u               /* CONFIG linear_num_key_heads */
#define QWEN36_GDN_VALUE_HEADS 48u             /* CONFIG linear_num_value_heads */
#define QWEN36_GDN_KEY_DIM 128u                /* CONFIG linear_key_head_dim */
#define QWEN36_GDN_VALUE_DIM 128u              /* CONFIG linear_value_head_dim */
#define QWEN36_GDN_CONV_KERNEL 4u              /* CONFIG linear_conv_kernel_dim */

// The GDN state plus the short causal convolution window it carries. Both are
// per-sequence and neither grows, so both live in one non-growing pool.
//
// THE STATE IS ONE PER VALUE HEAD: 48 of them, not 16. The reference form of
// gated DeltaNet (Qwen3-Next's modeling file, FLA's gated delta rule) repeats
// q and k from the 16 key heads to the 48 value heads and holds a 128x128
// state for every value head. This driver once sized and launched the
// recurrence as 16 states with three value heads "sharing" each - which read
// only value head 3h of every group, never touched 3h+1 and 3h+2, and wrote a
// 2048-wide output into the 6144-wide input of the output projection. A state
// per key head shared three ways is not GDN; the layer expands q and k and
// launches the delta rule at 48 heads.
//
// THE STATE IS FP32. LmDeltaRuleKernel's pool contract is four-byte elements -
// it addresses head h at h * KEY_DIM * VALUE_DIM * 4 and the K3 contract
// (K3_KDA_STATE_ELEMENT_BYTES = 4) says the same. This macro once sized the
// state at 2 bytes, which put the upper half of every sequence's heads past
// the end of the slot: at B1 the delta rule read and wrote beyond the pool.
// The element width is a named constant because the bf16-state option is a
// real, priced lever - it halves a state stream that is 310 MB/seq/token at
// this geometry (docs/PERF_ROADMAP_2026-08-01.md's K3 state correction is the
// same problem at twice the heads) - but it is a numerics question on a
// recurrence that compounds per token, and it lands as a bf16-pool delta-rule
// VARIANT with this constant flipped, never as this constant flipped alone.
// layer.cuh static_asserts the width against sizeof(float) so the flip cannot
// happen silently ahead of the kernel.
//
// The window is one tensor because the convolution runs on the fused QKV row
// before the split: QKV_DIM channels at KERNEL taps, bf16.
#define QWEN36_GDN_STATE_ELEMENT_BYTES 4u
#define QWEN36_GDN_STATE_BYTES \
	((QWEN36_GDN_VALUE_HEADS * QWEN36_GDN_KEY_DIM * QWEN36_GDN_VALUE_DIM \
		* QWEN36_GDN_STATE_ELEMENT_BYTES) \
	 + (QWEN36_GDN_QKV_DIM * QWEN36_GDN_CONV_KERNEL * 2u))

#define QWEN36_MTP_LAYERS 1u                   /* CONFIG mtp_num_hidden_layers */
#define QWEN36_KV_BITS 16u
#define QWEN36_KV_PAGE_SLOTS 64u

// -- layer kinds --
// 16 x (3 x gated DeltaNet -> 1 x gated full attention).
#define QWEN36_LAYER_KIND(layer) \
	(QWEN36_LAYER_IS_LINEAR(layer) ? LM_LAYER_RECURRENT : LM_LAYER_FULL)
