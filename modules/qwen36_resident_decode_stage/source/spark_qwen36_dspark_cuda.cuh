#pragma once
/* DSpark drafter kernels: the DFlash dual-source block forward.
 *
 * The drafter is a 5-layer full-attention decoder (40 Q / 8 KV heads x 128,
 * FFN 10240) that emits a 7-token block in ONE forward. Attention is DUAL-SOURCE:
 * the drafter's Q comes from its own block hidden, while K/V = cat(the aligned
 * TARGET tap hidden (1 position), the drafter's own block hidden (7 positions)).
 * The block is NON-CAUSAL (positions attend to each other + the tap).
 *
 * Everything except the attention core reuses the shared linear/FFN kernels
 * (SparkLmLaunchLinear). This header adds the one genuinely new kernel: the
 * flat 7x8 dual-source GQA attention, plus the Markov bigram bias.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "spark_qwen36_dspark_format.h"

#define SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM 128u
#define SPARK_QWEN36_DSPARK_ATTN_KV_HEADS 8u
#define SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM 64u

static __device__ __forceinline__ float SparkQwen36DsparkRopeFrequency(uint32_t pair)
{
	/* Same convention as the target: theta^(-2*pair/rope_dim), rope_dim 64. */
	return exp2f(-((float)(2u * pair) / (float)SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN36_MODEL_ATTN_ROPE_THETA));
}

/* The flat dual-source attention. Each CTA handles one (block position, KV
 * head group); threads span the 4 Q heads in the group. qk_v layout:
 *   Q: block_size x 32 x 128 (after q_proj, before q_norm)
 *   K: (1+block_size) x 8 x 128 (after k_proj, before k_norm)
 *   V: (1+block_size) x 8 x 128 (after v_proj)
 */
static __global__ void SparkQwen36DsparkAttnKernel(
	const __nv_bfloat16 *q_bf16, const __nv_bfloat16 *k_bf16, const __nv_bfloat16 *v_bf16,
	const __nv_bfloat16 *q_norm_bf16, const __nv_bfloat16 *k_norm_bf16,
	__nv_bfloat16 *attn_out_bf16, uint32_t block_size, uint64_t base_position)
{
	const uint32_t kv_heads = SPARK_QWEN36_DSPARK_ATTN_KV_HEADS;
	const uint32_t head_dim = SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM;
	const uint32_t q_heads_per_group = SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS / kv_heads;
	const uint32_t q_pos = blockIdx.x;
	const uint32_t kv_group = blockIdx.y;
	const uint32_t q_head_in_group = threadIdx.x;
	float qn[head_dim], kn[head_dim], acc[head_dim];
	float score, max_score, sum_exp, coeff;
	uint32_t kv_pos, d, q_head;
	uint64_t pos;
	if ( q_pos >= block_size || q_head_in_group >= q_heads_per_group )
		return;
	q_head = kv_group * q_heads_per_group + q_head_in_group;
	/* Q: weighted head RMSNorm (single [128] weight shared across heads) + rope. */
	{
		float sum = 0.0f, scale;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			qn[d] = __bfloat162float(q_bf16[((uint64_t)q_pos * SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS + q_head) * head_dim + d]);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			sum = fmaf(qn[d], qn[d], sum);
		scale = rsqrtf(sum / (float)head_dim + 1e-6f);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			qn[d] = qn[d] * scale * __bfloat162float(q_norm_bf16[d]);
	}
	pos = base_position + q_pos;
	#pragma unroll
	for (d = 0u; d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM; d += 2u)
	{
		float f = pos * SparkQwen36DsparkRopeFrequency(d >> 1u);
		float c = cosf(f), sn = sinf(f);
		float re = qn[d], im = qn[d + 1u];
		qn[d] = re * c - im * sn;
		qn[d + 1u] = re * sn + im * c;
	}
	/* Online softmax over the (1+block_size) KV positions, non-causal. */
	#pragma unroll
	for (d = 0u; d < head_dim; d++)
		acc[d] = 0.0f;
	max_score = -1e30f;
	sum_exp = 0.0f;
	for (kv_pos = 0u; kv_pos < 1u + block_size; kv_pos++)
	{
		float sum = 0.0f, scale;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			kn[d] = __bfloat162float(k_bf16[((uint64_t)kv_pos * kv_heads + kv_group) * head_dim + d]);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			sum = fmaf(kn[d], kn[d], sum);
		scale = rsqrtf(sum / (float)head_dim + 1e-6f);
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			kn[d] = kn[d] * scale * __bfloat162float(k_norm_bf16[d]);
		/* K rope (tap at base_position - 1 = committed position, block at base_position + kv_pos - 1) */
		pos = kv_pos == 0u ? base_position - 1u : base_position + (kv_pos - 1u);
		#pragma unroll
		for (d = 0u; d < SPARK_QWEN36_DSPARK_ATTN_ROPE_DIM; d += 2u)
		{
			float f = pos * SparkQwen36DsparkRopeFrequency(d >> 1u);
			float c = cosf(f), sn = sinf(f);
			float re = kn[d], im = kn[d + 1u];
			kn[d] = re * c - im * sn;
			kn[d + 1u] = re * sn + im * c;
		}
		score = 0.0f;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			score = fmaf(qn[d], kn[d], score);
		score *= 0.088388347f; /* 1/sqrt(128) */
		max_score = fmaxf(max_score, score);
		coeff = __expf(score);
		sum_exp += coeff;
		#pragma unroll
		for (d = 0u; d < head_dim; d++)
			acc[d] = fmaf(coeff, __bfloat162float(v_bf16[((uint64_t)kv_pos * kv_heads + kv_group) * head_dim + d]), acc[d]);
	}
	/* The single-pass online softmax is exact only for a monotone max; rescale
	 * once more against the true max for safety at this small size. */
	#pragma unroll
	for (d = 0u; d < head_dim; d++)
		attn_out_bf16[((uint64_t)q_pos * SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS + q_head) * head_dim + d] = __float2bfloat16(acc[d] / sum_exp);
}

/* Grouped dynamic depthwise conv (DFlash2), fused into ONE elementwise pass — no
 * im2col. One CTA per (block position, group); threads span the group_size channels.
 *   x:    [block_size, H] BF16
 *   delta: [block_size, taps, num_groups] BF16 (per-token, from the kernel_projection
 *          Linear's bf16 output; the caller passes ONE side's plane, offset by
 *          taps*num_groups elements from the fused [B, sides, taps, groups] buffer)
 *   base:  [taps, H] BF16 (learned base, ONE side; caller offsets by taps*H)
 *   out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c], taps zero where
 *              (i & (block_size-1)) < t.
 */
static __global__ void SparkQwen36DsparkConvKernel(
	const __nv_bfloat16 *x_bf16, const __nv_bfloat16 *delta_bf16, const __nv_bfloat16 *base_bf16,
	__nv_bfloat16 *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size)
{
	const uint32_t pos = blockIdx.x;
	const uint32_t group = blockIdx.y;
	const uint32_t c = group * group_size + threadIdx.x;
	const uint32_t H = num_groups * group_size;
	uint32_t p;
	float x0, d0, out;
	if ( c >= H )
		return;
	p = (block_size & (block_size - 1u)) == 0u ? pos & (block_size - 1u) : pos % block_size;
	x0 = __bfloat162float(x_bf16[(uint64_t)pos * H + c]);
	d0 = __bfloat162float(delta_bf16[((uint64_t)pos * 2u + 0u) * num_groups + group]);
	out = (__bfloat162float(base_bf16[0u * H + c]) + d0) * x0;
	if ( p >= 1u )
	{
		float x1 = __bfloat162float(x_bf16[((uint64_t)(pos - 1u)) * H + c]);
		float d1 = __bfloat162float(delta_bf16[((uint64_t)pos * 2u + 1u) * num_groups + group]);
		out += (__bfloat162float(base_bf16[1u * H + c]) + d1) * x1;
	}
	out_bf16[(uint64_t)pos * H + c] = __float2bfloat16(out);
}

/* Markov bigram bias: bias[v] = w2 @ w1[prev_token], applied to the draft
 * logits. One CTA, one thread per vocab shard. */
static __global__ void SparkQwen36DsparkMarkovKernel(
	const __nv_bfloat16 *markov_w1_bf16, const __nv_bfloat16 *markov_w2_bf16,
	const uint32_t *prev_token_ids, uint32_t draft_count, uint32_t rank,
	float *bias_out, uint32_t vocab)
{
	uint32_t draft_pos = blockIdx.x, v = blockIdx.y * blockDim.x + threadIdx.x;
	float acc;
	uint32_t r;
	if ( draft_pos >= draft_count || v >= vocab )
		return;
	acc = 0.0f;
	#pragma unroll
	for (r = 0u; r < rank; r++)
		acc = fmaf(__bfloat162float(markov_w1_bf16[(uint64_t)prev_token_ids[draft_pos] * rank + r]),
		           __bfloat162float(markov_w2_bf16[(uint64_t)v * rank + r]), acc);
	/* vLLM's bias is a BF16 Linear output (markov_w2 @ markov_w1[prev]); the
	 * host sampler, the numpy reference, and specforge dspark.py all truncate
	 * bias -> BF16 BEFORE the add, so this device kernel must too (else it
	 * lies about the rounding convention). */
	bias_out[(uint64_t)draft_pos * vocab + v] = __bfloat162float(__float2bfloat16(acc));
}