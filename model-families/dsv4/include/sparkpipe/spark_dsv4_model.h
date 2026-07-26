#ifndef SPARKPIPE_SPARK_DSV4_MODEL_H
#define SPARKPIPE_SPARK_DSV4_MODEL_H

#include <stdint.h>

/*
 * DeepSeek V4 Flash geometry, pinned against the checkpoint config
 * (huggingface.co/deepseek-ai/DeepSeek-V4-Flash config.json, fetched
 * 2026-07-19, transformers 4.57.1, model_type deepseek_v4). Every constant
 * below carries a CONFIG marker naming its source field. The Pro variant is
 * the same module with a second geometry header, the same trick as the
 * shared attention/MoE machinery it will reuse.
 *
 * Architecture in one paragraph: 43 layers over hidden 4096. The per-layer
 * attention kind comes from the config's compress_ratios array, embedded
 * below as the layer-kind table: the first two layers are sliding-window
 * attention (window 128), then compressed sparse attention (compress ratio
 * 4 with a 64-head x 128 indexer selecting top-512) alternates with
 * high-compression attention (ratio 128) through layer 42. All attention is
 * 64 query heads x head dim 512 over ONE key/value head, with a 1024-rank
 * query compression, rope on the first 64 dims (theta 10000 yarn-scaled x16
 * from 64K to 1M; the compressed stream has its own theta 160000), and the
 * output projection grouped 8 ways. MoE everywhere the reference says so:
 * 256 routed experts (fp4 native) top-6 with sqrtsoftplus scoring and the
 * noaux_tc balancer, one shared expert, expert intermediate 2048, swiglu
 * clamp 10, routed scale 1.5, and the first three layers hash-routed. The
 * residual stream is hyper-connected: hc_mult 4 streams cross every stage
 * boundary (Sinkhorn iterations are the training-side mixer). One MTP
 * layer. Vocabulary 129280, untied. FP8 e4m3 with ue8m0 scales outside the
 * experts.
 *
 * REFERENCE-PIN RESOLVED (against inference/model.py + kernel.py +
 * convert.py, fetched 2026-07-19, identical across both repos):
 * - compress_ratios carries n_layers+1 entries; the extra final 0 is the
 *   MTP layer: window-only attention at base theta, YaRN off.
 * - Every layer is MoE - no dense layers exist, no first_k_dense_replace;
 *   layers 0..2 route by the checkpoint's tid2eid[vocab, topk] int32 table
 *   (no balancer bias tensor there); all others score-route.
 * - Per-layer cache = window ring of 128 head_dim entries (slot pos%128)
 *   plus, on compress layers, an append-only compressed stream at
 *   pos/ratio; entries are kv_norm'd, rope'd on the last 64 dims, non-rope
 *   dims fp8-sim quantized in blocks of 64 with power-of-two scales. The
 *   ratio-4 compressor OVERLAPS: doubled channels pool the previous group
 *   through the first channel half. A compress-bearing layer runs theta
 *   160000 YaRN-scaled for BOTH its streams; a ratio-0 layer runs theta
 *   10000 unscaled - the split is per LAYER, not per stream.
 * - o composition: heads*512 viewed as o_groups groups of 4096, per-group
 *   einsum against wo_a[group][1024, 4096] (bf16 after convert), groups*1024
 *   concatenated into wo_b. Queries take an unweighted per-head rms before
 *   rope; the sink joins the softmax denominator only; the output is
 *   INVERSE-rotated on its last 64 dims.
 * - sqrtsoftplus router: scores = sqrt(softplus(z)); top-k selects on
 *   scores + bias but weights gather ORIGINAL scores, sum-normalize, and
 *   scale by routed_scaling_factor (noaux_tc).
 * - mHC runs Sinkhorn AT INFERENCE: 20 iterations on the 4x4 comb after a
 *   row softmax, with +eps after the softmax and inside every
 *   normalization; pre = sigmoid(m*s0+b)+eps, post = 2*sigmoid(m*s1+b);
 *   the head reduction is the sigmoid pre-form only. hc_post writes stream
 *   k as post[k]*out + sum_j comb[j][k]*residual[j] (comb transposed).
 * - swiglu clamp: up to [-limit, limit], gate to max limit only; the
 *   routing weight multiplies the fp32 intermediate before w2.
 * - Indexer: q = wq_b(shared q_lora), rope, Hadamard (scale d^-0.5),
 *   fp4-sim block 32; per-head weights = weights_proj(x) * d^-0.5 *
 *   heads^-0.5; score = sum_h relu(q_h . kv_c) * w_h over its OWN
 *   128-dim rotated compressed cache; top-k indices, +window offset.
 */

#define SPARK_DSV4_MODEL_HIDDEN_DIMENSION 4096u                 /* CONFIG hidden_size */
#define SPARK_DSV4_MODEL_LAYER_COUNT 43u                        /* CONFIG num_hidden_layers */
#define SPARK_DSV4_MODEL_VOCAB_COUNT 129280u                    /* CONFIG vocab_size */
#define SPARK_DSV4_MODEL_MAX_POSITIONS 1048576u                 /* CONFIG max_position_embeddings */
#define SPARK_DSV4_MODEL_RMS_NORM_EPSILON 1e-6f                 /* CONFIG rms_norm_eps */
#define SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT 64u              /* CONFIG num_attention_heads */
#define SPARK_DSV4_MODEL_ATTN_KV_HEAD_COUNT 1u                  /* CONFIG num_key_value_heads */
#define SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION 512u               /* CONFIG head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION 64u                /* CONFIG qk_rope_head_dim */
#define SPARK_DSV4_MODEL_ATTN_ROPE_THETA 10000.0f               /* CONFIG rope_theta */
#define SPARK_DSV4_MODEL_ATTN_YARN_FACTOR 16u                   /* CONFIG rope_scaling.factor */
#define SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS 65536u    /* CONFIG rope_scaling.original */
#define SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA 160000.0f          /* CONFIG compress_rope_theta */
#define SPARK_DSV4_MODEL_QUERY_LORA_RANK 1024u                  /* CONFIG q_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_LORA_RANK 1024u                 /* CONFIG o_lora_rank */
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT 8u                  /* CONFIG o_groups */
#define SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS 128u             /* CONFIG sliding_window */
#define SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO 4u                  /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO 128u                /* CONFIG compress_ratios */
#define SPARK_DSV4_MODEL_INDEX_HEAD_COUNT 64u                   /* CONFIG index_n_heads */
#define SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION 128u              /* CONFIG index_head_dim */
#define SPARK_DSV4_MODEL_INDEX_TOP_K 512u                       /* CONFIG index_topk */
#define SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT 256u               /* CONFIG n_routed_experts */
#define SPARK_DSV4_MODEL_SHARED_EXPERT_COUNT 1u                 /* CONFIG n_shared_experts */
#define SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN 6u                   /* CONFIG num_experts_per_tok */
#define SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION 2048u    /* CONFIG moe_intermediate_size */
#define SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT 3u             /* CONFIG num_hash_layers */
#define SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR 1.5f             /* CONFIG routed_scaling_factor */
#define SPARK_DSV4_MODEL_SWIGLU_LIMIT 10.0f                     /* CONFIG swiglu_limit */
#define SPARK_DSV4_MODEL_HC_STREAM_COUNT 4u                     /* CONFIG hc_mult */
#define SPARK_DSV4_MODEL_MTP_LAYER_COUNT 1u                     /* CONFIG num_nextn_predict_layers */
#define SPARK_DSV4_MODEL_ROPE_BETA_FAST 32u                     /* CONFIG beta_fast */
#define SPARK_DSV4_MODEL_ROPE_BETA_SLOW 1u                      /* CONFIG beta_slow */
#define SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS 20u             /* CONFIG hc_sinkhorn_iters, RUNS AT INFERENCE */
#define SPARK_DSV4_MODEL_HC_EPSILON 1e-6f                       /* CONFIG hc_eps */
#define SPARK_DSV4_MODEL_MTP_LAYER_KIND SPARK_DSV4_MODEL_LAYER_KIND_SWA /* compress_ratios[n_layers] == 0 */
#define SPARK_DSV4_MODEL_KV_QUANT_BLOCK 64u                     /* act_quant(kv nope dims, 64) */
#define SPARK_DSV4_MODEL_ACT_QUANT_BLOCK 128u                   /* linear activation quant group */
#define SPARK_DSV4_MODEL_FP4_QUANT_BLOCK 32u                    /* fp4_block_size */
#define SPARK_DSV4_MODEL_FP8_MAX 448.0f
#define SPARK_DSV4_MODEL_FP4_MAX 6.0f
#define SPARK_DSV4_MODEL_QUANT_AMAX_FLOOR 1e-4f
#define SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR 2u                  /* ratio-4 compressor doubles channels */
#define SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES 2u

#define SPARK_DSV4_MODEL_HC_MIX_ROWS ((2u + SPARK_DSV4_MODEL_HC_STREAM_COUNT) * SPARK_DSV4_MODEL_HC_STREAM_COUNT)

#define SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION (SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT)
#define SPARK_DSV4_MODEL_INDEX_DIMENSION (SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION)
#define SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS (SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HIDDEN_DIMENSION)

#define SPARK_DSV4_MODEL_LAYER_KIND_SWA 0u
#define SPARK_DSV4_MODEL_LAYER_KIND_CSA 1u
#define SPARK_DSV4_MODEL_LAYER_KIND_HCA 2u

// The per-layer attention map, transcribed from CONFIG compress_ratios
// (0 -> SWA, 4 -> CSA, 128 -> HCA); the array's 44th entry belongs to the
// MTP layer and is a REFERENCE-PIN. Static per translation unit; the two
// consumers are the pack format and the module.
static const uint8_t SPARK_DSV4_MODEL_LAYER_KIND[SPARK_DSV4_MODEL_LAYER_COUNT] =
{
	0,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
	1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1
};

static inline uint32_t SparkDsv4ModelLayerKind(uint32_t layer_index)
{
	return(layer_index < SPARK_DSV4_MODEL_LAYER_COUNT ? (uint32_t)SPARK_DSV4_MODEL_LAYER_KIND[layer_index] : SPARK_DSV4_MODEL_LAYER_KIND_SWA);
}

#endif
